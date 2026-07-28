/*
 * mset_visualizer.h
 * ==================
 *
 * Public API for mset_visualizer.c - see that file for details.
 *
 * Renders the same content as mset_visualizer.py's MSETAnomalyVisualizer as a
 * single self-contained, dependency-free HTML file: a rotatable/zoomable 3D
 * scatter of input states, MSET estimates and SPRT-flagged anomalies, with a
 * linked SPRT panel underneath. No display server is required to *produce*
 * the file - open it in any browser to interact with it.
 */
#ifndef MSET_VISUALIZER_H
#define MSET_VISUALIZER_H

#include "mset_rul.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double alpha;          /* SPRT false-alarm rate (default 0.01)          */
    double beta;            /* SPRT missed-detection rate (default 0.01)     */
    double disturbance;     /* SPRT shift magnitude in sigmas (default 3.0)  */
    const char *title;      /* page title; NULL for a sensible default       */
    const char *time_label; /* x-axis label; NULL defaults to "Time"         */
} MSETVisualizerOptions;

/* Fills an options struct with the library defaults. */
void mset_visualizer_default_options(MSETVisualizerOptions *opt);

/*
 * Write a self-contained interactive HTML visualization to `path`.
 *
 *   model         - a fitted MSET model (mset_fit already called).
 *   observations  - k x model->n_params monitored states, row-major.
 *   times         - k time values (e.g. days); NULL defaults to 1..k.
 *   k             - number of observations (must be >= 2).
 *   param_names   - model->n_params display names; NULL defaults to p1, p2, ...
 *   opt           - rendering / SPRT options; NULL uses the defaults.
 *
 * Requires model->n_params >= 3 (the 3D chart needs three axes; only the
 * first three parameters are plotted in 3D, but every parameter gets its own
 * SPRT channel in the lower panel, exactly as in the Python version).
 *
 * Returns 0 on success, -1 on error (bad arguments or file I/O failure).
 */
int mset_visualize_write_html(const char *path,
                              const MSET *model,
                              const double *observations,
                              const double *times,
                              int k,
                              const char * const *param_names,
                              const MSETVisualizerOptions *opt);

#ifdef __cplusplus
}
#endif

#endif /* MSET_VISUALIZER_H */
