/*
 * csv_io.h
 * ========
 *
 * A small, dependency-free (libc-only) CSV reader that adapts to any size
 * input: row and column counts come entirely from the file's content, never
 * from a compile-time constant. This is what makes mset_rul/mset_visualizer
 * usable with real telemetry of any width (3 parameters, 5, 40, whatever the
 * file has) instead of the fixed 3-parameter synthetic demo data.
 *
 * The file must be rectangular - every data row needs the same number of
 * fields as the first data row - since that's what MSET's matrix math
 * requires; a ragged file is reported as a parse error with a line number
 * rather than silently truncated or padded.
 */
#ifndef CSV_IO_H
#define CSV_IO_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double *data;     /* rows x cols, row-major, malloc'd                  */
    int rows;          /* data rows (excludes the header line, if any)      */
    int cols;          /* columns, taken from the first data row            */
    char **headers;    /* cols column names if has_header was set, else NULL */
} CsvMatrix;

/*
 * Read a delimited numeric matrix from `path`.
 *
 *   has_header : if non-zero, the first non-blank line is column names
 *                (captured in out->headers) rather than data.
 *   delimiter  : field separator, e.g. ',' for CSV or '\t' for TSV.
 *
 * Fields may optionally be wrapped in a single pair of double quotes (enough
 * for typical spreadsheet-exported CSVs); a single trailing empty field from
 * a trailing delimiter is dropped. Blank lines are skipped.
 *
 * On success, fills `out` (caller must csv_free() it) and returns 0. On
 * failure, returns -1 and leaves `out` untouched; a diagnostic naming the
 * file and line number is printed to stderr.
 */
int csv_read(const char *path, int has_header, char delimiter, CsvMatrix *out);

/* Convenience wrapper: csv_read(path, has_header, ',', out). */
int csv_read_matrix(const char *path, int has_header, CsvMatrix *out);

/*
 * Case-insensitive lookup of a column by header name. Returns the 0-based
 * column index, or -1 if `m` has no headers or none match.
 */
int csv_find_column(const CsvMatrix *m, const char *name);

/*
 * Remove column `col_index` from `m` in place (m->cols shrinks by one,
 * m->data and m->headers are reallocated to match) and return its values
 * separately via a newly malloc'd array of m->rows doubles (caller frees).
 * Returns 0 on success, -1 if col_index is out of range.
 */
int csv_extract_column(CsvMatrix *m, int col_index, double **out_column);

/* Frees data/headers and zeroes the struct. Safe to call on a zeroed or
 * already-freed CsvMatrix. */
void csv_free(CsvMatrix *m);

/*
 * ---- The two MSET-facing loaders -------------------------------------
 *
 * Both adapt to whatever number of rows and columns the file contains -
 * there is no fixed-size limit and no compile-time parameter count.
 */

/*
 * Load healthy training data: every column is a monitored parameter, every
 * row is one healthy historic state (Eq. 2's training matrix T in
 * mset_rul.c). No time column is expected - training states don't need a
 * timestamp.
 */
int csv_load_training(const char *path, int has_header, CsvMatrix *out);

/*
 * Load monitored/test data: same rectangular format as csv_load_training,
 * plus optional automatic time-column extraction.
 *
 * If `time_col_candidates` is a NULL-terminated array of candidate header
 * names (case-insensitive, e.g. {"time","day","days","t",NULL}), the first
 * one found among the file's columns is pulled out of the returned matrix
 * and returned separately via `out_times` (out->cols shrinks by one to
 * match csv_load_training's column count). Pass NULL for
 * `time_col_candidates` - or leave every candidate unmatched - to keep every
 * column as data; `*out_times` is then set to NULL, and the caller can
 * default the time axis to 1..rows (mset_visualize_write_html already does
 * this when given a NULL times pointer).
 *
 * Caller frees `*out_times` with free() (may be NULL).
 */
int csv_load_test(const char *path, int has_header,
                  const char * const *time_col_candidates,
                  CsvMatrix *out, double **out_times);

#ifdef __cplusplus
}
#endif

#endif /* CSV_IO_H */
