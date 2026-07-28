/*
 * mset_demo_viz.c
 * ================
 *
 * Demonstration program: fits an MSET model on synthetic healthy data,
 * monitors a unit that drifts into failure, and writes an interactive HTML
 * visualization (mset_visualizer.c) of the input, the MSET estimates and the
 * SPRT-flagged anomalies.
 *
 * Build:
 *   gcc -std=c99 -O2 -Wall -Wextra -c mset_rul.c -DMSET_RUL_NO_MAIN -o mset_rul.o
 *   gcc -std=c99 -O2 -Wall -Wextra -c mset_visualizer.c -o mset_visualizer.o
 *   gcc -std=c99 -O2 -Wall -Wextra -c mset_demo_viz.c -o mset_demo_viz.o
 *   gcc mset_rul.o mset_visualizer.o mset_demo_viz.o -lm -o mset_demo_viz
 *
 * Run:
 *   ./mset_demo_viz [output.html]
 *
 * Then open the HTML file in any browser.
 */

#include <stdio.h>
#include <stdlib.h>

#include "mset_rul.h"
#include "mset_visualizer.h"

int main(int argc, char **argv)
{
    const char *out_path = (argc > 1) ? argv[1] : "mset_visualization.html";

    Rng rng;
    rng_init(&rng, 20070101ULL);

    /* healthy training history */
    enum { TRAIN_DAYS = 60 };
    double *training = malloc(sizeof(double) * TRAIN_DAYS * DEMO_PARAMS);
    generate_unit(training, TRAIN_DAYS, 0.0, &rng, 0.35);

    MSET *model = mset_create(kernel_inverse_distance, 0.30, 1e-8, 1);
    if (mset_fit(model, training, TRAIN_DAYS, DEMO_PARAMS, 14) != 0) {
        fprintf(stderr, "mset_fit failed\n");
        return 1;
    }

    /* monitored unit: healthy for 14 days, then drifting for 22 */
    enum { HEALTHY_DAYS = 14, DRIFT_DAYS = 22 };
    enum { MON_DAYS = HEALTHY_DAYS + DRIFT_DAYS };
    double *monitored = malloc(sizeof(double) * MON_DAYS * DEMO_PARAMS);
    generate_unit(monitored, HEALTHY_DAYS, 0.0005, &rng, 0.35);
    generate_unit(monitored + (size_t)HEALTHY_DAYS * DEMO_PARAMS,
                  DRIFT_DAYS, 0.030, &rng, 0.35);

    double times[MON_DAYS];
    for (int i = 0; i < MON_DAYS; ++i) times[i] = i + 1;

    const char *param_names[DEMO_PARAMS] = {
        "Temperature [degC]", "Rel. humidity [%]", "Vibration [g]"
    };

    MSETVisualizerOptions opt;
    mset_visualizer_default_options(&opt);
    opt.alpha = 0.01;
    opt.beta = 0.01;
    opt.disturbance = 3.0;

    int rc = mset_visualize_write_html(out_path, model, monitored, times,
                                       MON_DAYS, param_names, &opt);
    if (rc != 0) {
        fprintf(stderr, "mset_visualize_write_html failed\n");
        mset_free(model);
        free(training);
        free(monitored);
        return 1;
    }

    printf("Wrote %s - open it in a browser.\n", out_path);
    printf("(training states=%d, memory matrix=%d, monitored samples=%d)\n",
          TRAIN_DAYS, model->n_memory, MON_DAYS);

    mset_free(model);
    free(training);
    free(monitored);
    return 0;
}
