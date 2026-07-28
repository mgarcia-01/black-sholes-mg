/*
 * mset_demo_csv.c
 * ================
 *
 * The CSV-driven workflow: fit an MSET model from a training CSV of any
 * size, monitor a test CSV of any size, and render the interactive HTML
 * visualization (mset_visualizer.c). Row and column counts come entirely
 * from the two files - nothing here is fixed to a particular parameter
 * count the way the synthetic demo (mset_demo_viz.c) is.
 *
 * CSV format: one optional header row of column names, then one row per
 * sample/state, one column per monitored parameter - see csv_io.h. The
 * training file should contain only healthy states and no time column. The
 * test file may include a time column named "time", "day", "days", "t", or
 * "index" (case-insensitive); if present it's used as the time axis instead
 * of defaulting to 1..k, and it's excluded from the parameter columns.
 *
 * Usage:
 *   ./mset_demo_csv <training.csv> <test.csv> [output.html] [memory_size]
 *
 * training.csv and test.csv must have the same number of parameter columns
 * (after any time column is removed from test.csv). The visualizer needs at
 * least 3 parameter columns, since a 3D chart has three axes.
 *
 * Build:
 *   gcc -std=c99 -O2 -Wall -Wextra -c mset_rul.c -DMSET_RUL_NO_MAIN -o mset_rul.o
 *   gcc -std=c99 -O2 -Wall -Wextra -c mset_visualizer.c -o mset_visualizer.o
 *   gcc -std=c99 -O2 -Wall -Wextra -c csv_io.c -o csv_io.o
 *   gcc -std=c99 -O2 -Wall -Wextra -c mset_demo_csv.c -o mset_demo_csv.o
 *   gcc mset_rul.o mset_visualizer.o csv_io.o mset_demo_csv.o -lm -o mset_demo_csv
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "csv_io.h"
#include "mset_rul.h"
#include "mset_visualizer.h"

static const char *TIME_COLUMN_CANDIDATES[] = {
    "time", "day", "days", "t", "index", NULL
};

/* Builds a NULL-terminated-free array of `const char *` views into an
 * already-owned CsvMatrix's headers, for passing as param_names (no copy,
 * no extra ownership - valid as long as the CsvMatrix is alive). */
static const char **header_view(const CsvMatrix *m)
{
    if (!m->headers) return NULL;
    const char **view = malloc(sizeof(char *) * (size_t)m->cols);
    if (!view) return NULL;
    for (int j = 0; j < m->cols; ++j) view[j] = m->headers[j];
    return view;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
               "usage: %s <training.csv> <test.csv> [output.html] [memory_size]\n",
               argv[0]);
        return 1;
    }
    const char *train_path = argv[1];
    const char *test_path = argv[2];
    const char *out_path = (argc > 3) ? argv[3] : "mset_visualization.html";
    int memory_size = (argc > 4) ? atoi(argv[4]) : 0;   /* 0 -> adaptive default */

    /* ---- load training data (any rows x any columns) -------------------- */
    CsvMatrix train = {0};
    if (csv_load_training(train_path, /*has_header=*/1, &train) != 0) {
        fprintf(stderr, "failed to load training data from '%s'\n", train_path);
        return 1;
    }
    printf("training : %-28s %4d rows x %d columns\n", train_path, train.rows, train.cols);

    /* ---- load test/monitoring data, auto-detecting a time column -------- */
    CsvMatrix test = {0};
    double *times = NULL;
    if (csv_load_test(test_path, /*has_header=*/1, TIME_COLUMN_CANDIDATES,
                      &test, &times) != 0) {
        fprintf(stderr, "failed to load test data from '%s'\n", test_path);
        csv_free(&train);
        return 1;
    }
    printf("test     : %-28s %4d rows x %d columns%s\n",
          test_path, test.rows, test.cols, times ? " (+ time column)" : "");

    /* ---- validate shapes -------------------------------------------------*/
    if (train.cols != test.cols) {
        fprintf(stderr,
               "column count mismatch: training has %d parameter columns, "
               "test has %d (after removing any time column) - both files "
               "must monitor the same parameters\n",
               train.cols, test.cols);
        free(times);
        csv_free(&train);
        csv_free(&test);
        return 1;
    }
    if (test.cols < 3) {
        fprintf(stderr,
               "the visualizer needs at least 3 parameter columns (a 3D chart "
               "has three axes); '%s' has %d\n", test_path, test.cols);
        free(times);
        csv_free(&train);
        csv_free(&test);
        return 1;
    }

    /* ---- fit MSET on the training matrix ---------------------------------*/
    MSET *model = mset_create(kernel_inverse_distance, 0.30, 1e-8, /*normalize=*/1);
    if (mset_fit(model, train.data, train.rows, train.cols, memory_size) != 0) {
        fprintf(stderr, "mset_fit failed\n");
        mset_free(model);
        free(times);
        csv_free(&train);
        csv_free(&test);
        return 1;
    }
    printf("MSET     : memory matrix D = %d states (from %d training states)\n",
          model->n_memory, train.rows);

    /* ---- times: use the extracted column, or default to 1..k ------------ */
    double *own_times = NULL;
    const double *time_values = times;
    if (!time_values) {
        own_times = malloc(sizeof(double) * (size_t)test.rows);
        for (int i = 0; i < test.rows; ++i) own_times[i] = i + 1;
        time_values = own_times;
    }

    /* ---- param names from whichever file has headers --------------------*/
    const char **param_names = header_view(&test);
    if (!param_names) param_names = header_view(&train);

    MSETVisualizerOptions opt;
    mset_visualizer_default_options(&opt);

    int rc = mset_visualize_write_html(out_path, model, test.data, time_values,
                                       test.rows, param_names, &opt);
    if (rc != 0) {
        fprintf(stderr, "mset_visualize_write_html failed\n");
    } else {
        printf("wrote    : %s - open it in a browser\n", out_path);
    }

    free(param_names);
    free(own_times);
    free(times);
    mset_free(model);
    csv_free(&train);
    csv_free(&test);
    return rc == 0 ? 0 : 1;
}
