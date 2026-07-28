/*
 * mset_rul.h
 * ==========
 *
 * Public API of mset_rul.c - see that file for full documentation of the
 * algorithms and the Cheng & Pecht (2007) equations they implement.
 *
 * Build mset_rul.c with -DMSET_RUL_NO_MAIN to omit its own main() and link it
 * as a library into another program (e.g. the visualizer demo), or compile
 * it without that flag to get the original standalone console demo.
 */
#ifndef MSET_RUL_H
#define MSET_RUL_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- similarity kernels: the paper's unspecified "(x)" operator -------- */

typedef double (*mset_kernel_fn)(const double *u, const double *v, int n,
                                 double param);

double kernel_inverse_distance(const double *u, const double *v, int n, double scale);
double kernel_gaussian(const double *u, const double *v, int n, double bandwidth);
double kernel_bounded_angle(const double *u, const double *v, int n, double unused);

/* Eq. (10): Euclidean norm of a residual (or any) vector. */
double euclidean_norm(const double *v, int n);

/* ---- MSET engine --------------------------------------------------- */

typedef struct {
    int n_params;             /* n  - monitored parameters                  */
    int n_memory;              /* m  - states in memory matrix D (Eq. 3)     */
    int n_remaining;           /* |L| - remaining training data (Eq. 4)      */

    mset_kernel_fn kernel;
    double kernel_param;
    double ridge;
    int normalize;

    double *lo, *span;        /* per-parameter min-max scaler (n)           */
    double *D_states;         /* m x n, rows = memory states (scaled)       */
    double *L_states;         /* |L| x n, remaining training data (scaled)  */
    int *D_index, *L_index;   /* indices into the original training set     */
    double *D_matrix;         /* n x m: columns are memory states (Eq. 3)   */
    double *G_inv;            /* m x m: (D^T (x) D)^-1  (Eq. 7)             */
    int fitted;
} MSET;

MSET *mset_create(mset_kernel_fn kernel, double kernel_param, double ridge,
                  int normalize);
void mset_free(MSET *m);
int mset_fit(MSET *m, const double *training, int l, int n, int memory_size);

int mset_estimate(const MSET *m, const double *observation, double *est);
int mset_residual(const MSET *m, const double *observation, double *res,
                  int scaled);
int mset_residuals(const MSET *m, const double *observations, int k,
                   double *residuals, int scaled);
int mset_healthy_residuals(const MSET *m, double *out);
int mset_healthy_statistics(const MSET *m, double *mu, double *sd);

/* ---- fault detection: two-sided SPRT (Fig. 2) ------------------------ */

typedef enum { SPRT_CONTINUE = 0, SPRT_HEALTHY = 1, SPRT_FAULT = 2 } SprtVerdict;

typedef struct {
    double mu0, sigma, shift;
    double upper, lower;
    double llr_pos, llr_neg;
} SPRT;

void sprt_init(SPRT *s, double mean, double sigma, double alpha, double beta,
               double disturbance);
SprtVerdict sprt_update(SPRT *s, double value);
int sprt_first_fault(SPRT *s, const double *series, int k);

/* ---- degradation model (Eq. 9, 10, 11) -------------------------------- */

void accumulated_degradation(const double *residuals, int k, int n, double *out);
void degradation(const double *residuals, int k, int n, double *out);

/* ---- criteria of failure (Fig. 3) -------------------------------------- */

typedef struct {
    double mean, sigma, k_sigma;
    double lower, upper;
} FailureCriteria;

int failure_criteria_init(FailureCriteria *c, const double *failure_degradations,
                          int count, double k_sigma);

/* ---- degradation regression and RUL prediction (Fig. 3, Figs. 5-6) ---- */

typedef struct {
    double slope, intercept, r_squared;
} LinearTrend;

int linear_trend_fit(LinearTrend *t, const double *times, const double *values,
                     int k);
double linear_trend_predict(const LinearTrend *t, double time);
double linear_trend_time_to_reach(const LinearTrend *t, double level);

typedef struct {
    double a, b, r_squared;
} PowerTrend;

int power_trend_fit(PowerTrend *p, const double *times, const double *values,
                    int k);
double power_trend_time_to_reach(const PowerTrend *p, double level);

typedef struct {
    double current_time;
    double failure_time_lower, failure_time_nominal, failure_time_upper;
    double rul_lower, rul_nominal, rul_upper;
    LinearTrend trend;
    FailureCriteria criteria;
} RULPrediction;

int predict_rul(RULPrediction *out, const double *times,
                const double *degradation_series, int k,
                const FailureCriteria *criteria);

/* ---- deterministic PRNG + synthetic data generator (used by demos) ---- */

typedef struct { unsigned long long s; int has_gauss; double gauss; } Rng;

void rng_init(Rng *r, unsigned long long seed);
double rng_uniform(Rng *r);
double rng_gauss(Rng *r, double mu, double sigma);

#define DEMO_PARAMS 3
void generate_unit(double *out, int days, double drift_rate, Rng *rng,
                   double noise);

#ifdef __cplusplus
}
#endif

#endif /* MSET_RUL_H */
