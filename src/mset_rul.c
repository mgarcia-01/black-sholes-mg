/*
 * mset_rul.c
 * ==========
 *
 * C (C99) implementation, dependency-free (libc + libm only), of:
 *
 *     S. Cheng and M. Pecht, "Multivariate State Estimation Technique for
 *     Remaining Useful Life Prediction of Electronic Products",
 *     CALCE, University of Maryland (AAAI 2007).
 *
 * This is a direct port of mset_rul.py - same structure, same algorithms,
 * same equation coverage:
 *
 *     Eq. (1)   Observation / state vector          X(ti) = [x1i, ..., xni]^T
 *     Eq. (2)   Training data matrix                T = [X(t1), ..., X(tl)]
 *     Eq. (3)   Memory matrix                       D = [X1, ..., Xm]
 *     Eq. (4)   Remaining training data             L,  T = D u L
 *     Eq. (5)   Estimate                            Xest = D . W
 *     Eq. (6)   Error vector                        eps = Xobs - Xest
 *     Eq. (7)   Weight vector (least squares)       W = (D^T (x) D)^-1 . (D^T (x) Xobs)
 *     Eq. (8)   Estimate in closed form             Xest = D . (D^T (x) D)^-1 . (D^T (x) Xobs)
 *     Fig. 2    Full MSET process (D/L split, healthy residuals, actual
 *               residuals) + two-sided SPRT for fault detection
 *     Eq. (9)   Accumulated degradation             De_Ac(tk) = SUM_i ||R(ti)||
 *     Eq. (10)  Residual Euclidean norm             ||R(ti)|| = sqrt(SUM_j r_ji^2)
 *     Eq. (11)  Sample-count-normalised degradation De(tk) = (1/k) SUM_i ||R(ti)||
 *     Fig. 3    RUL prediction model (criteria of failure from failed units /
 *               accelerated tests, degradation regression, RUL prediction)
 *
 * As in the Python version, the paper's unspecified nonlinear operator "(x)"
 * is a pluggable similarity kernel (function pointer); inverse-distance,
 * Gaussian and bounded-angle operators are provided, inverse-distance being
 * the classical MSET default.
 *
 * Build and run:
 *     gcc -std=c99 -O2 -Wall -Wextra -o mset_rul mset_rul.c -lm
 *     ./mset_rul
 *
 * The executable runs the same two programs as `python3 mset_rul.py`:
 *   1. a synthetic end-to-end demonstration (training -> accelerated test ->
 *      criteria of failure -> monitored unit -> SPRT -> RUL);
 *   2. a numerical reproduction of the paper's published case study
 *      (Table 1, Figures 5-6).
 *
 * Memory conventions: every *_create() has a matching *_free(); matrices are
 * row-major double arrays; a "state" is a double[n_params].
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Optional API surface: kept in this single-file library even when the demo
 * does not call it, without triggering -Wunused-function. */
#if defined(__GNUC__) || defined(__clang__)
#define MSET_API static __attribute__((unused))
#else
#define MSET_API static
#endif

/* ======================================================================== */
/* 1. Minimal linear algebra (no BLAS/LAPACK)                               */
/* ======================================================================== */

/* Eq. (10): Euclidean norm of a residual (or any) vector. */
static double euclidean_norm(const double *v, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += v[i] * v[i];
    return sqrt(s);
}

static double euclidean_distance(const double *u, const double *v, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = u[i] - v[i];
        s += d * d;
    }
    return sqrt(s);
}

/* y(p) = A(p x q) . x(q), row-major A */
static void matvec(const double *a, const double *x, double *y, int p, int q)
{
    for (int i = 0; i < p; ++i) {
        double s = 0.0;
        for (int j = 0; j < q; ++j) s += a[(size_t)i * q + j] * x[j];
        y[i] = s;
    }
}

/*
 * Invert a square n x n matrix in place of `out` by Gauss-Jordan elimination
 * with partial pivoting. `ridge` adds lambda*I before inversion (Tikhonov
 * regularisation): the similarity matrix D^T (x) D of Eq. (7) is frequently
 * ill-conditioned when memory-matrix states crowd together, and the ridge is
 * the practical safeguard against numerical blow-up.
 *
 * Returns 0 on success, -1 if singular to working precision.
 */
static int invert(const double *a, double *out, int n, double ridge)
{
    size_t w = (size_t)(2 * n);
    double *aug = malloc(sizeof(double) * (size_t)n * w);
    if (!aug) return -1;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            aug[i * w + j] = a[(size_t)i * n + j] + (i == j ? ridge : 0.0);
            aug[i * w + n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int col = 0; col < n; ++col) {
        /* partial pivoting */
        int pivot_row = col;
        double best = fabs(aug[(size_t)col * w + col]);
        for (int r = col + 1; r < n; ++r) {
            double v = fabs(aug[(size_t)r * w + col]);
            if (v > best) { best = v; pivot_row = r; }
        }
        if (best < 1e-15) { free(aug); return -1; }
        if (pivot_row != col) {
            for (size_t j = 0; j < w; ++j) {
                double t = aug[(size_t)col * w + j];
                aug[(size_t)col * w + j] = aug[(size_t)pivot_row * w + j];
                aug[(size_t)pivot_row * w + j] = t;
            }
        }

        double pivot = aug[(size_t)col * w + col];
        for (size_t j = 0; j < w; ++j) aug[(size_t)col * w + j] /= pivot;

        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            double f = aug[(size_t)r * w + col];
            if (f != 0.0)
                for (size_t j = 0; j < w; ++j)
                    aug[(size_t)r * w + j] -= f * aug[(size_t)col * w + j];
        }
    }

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            out[(size_t)i * n + j] = aug[(size_t)i * w + n + j];
    free(aug);
    return 0;
}

/* ======================================================================== */
/* 2. Similarity operators - the "(x)" of Eq. (7) / Eq. (8)                 */
/* ======================================================================== */

typedef double (*mset_kernel_fn)(const double *u, const double *v, int n,
                                 double param);

/* s(u,v) = 1 / (1 + d(u,v)/scale) - the classical MSET operator. */
static double kernel_inverse_distance(const double *u, const double *v, int n,
                                      double scale)
{
    return 1.0 / (1.0 + euclidean_distance(u, v, n) / scale);
}

/* s(u,v) = exp(-(d/h)^2) - smoother, sharper locality. */
MSET_API double kernel_gaussian(const double *u, const double *v, int n,
                              double bandwidth)
{
    double d = euclidean_distance(u, v, n) / bandwidth;
    return exp(-d * d);
}

/* s(u,v) = 1 - theta/(pi/2), clipped to [0,1] - direction sensitive. */
MSET_API double kernel_bounded_angle(const double *u, const double *v, int n,
                                   double unused)
{
    (void)unused;
    double nu = euclidean_norm(u, n), nv = euclidean_norm(v, n);
    if (nu == 0.0 || nv == 0.0) return (nu == nv) ? 1.0 : 0.0;
    double dot = 0.0;
    for (int i = 0; i < n; ++i) dot += u[i] * v[i];
    double c = dot / (nu * nv);
    if (c > 1.0) c = 1.0;
    if (c < -1.0) c = -1.0;
    double s = 1.0 - acos(c) / (M_PI / 2.0);
    return s > 0.0 ? s : 0.0;
}

/* ======================================================================== */
/* 3. MSET engine                                                           */
/* ======================================================================== */

typedef struct {
    int n_params;            /* n  - monitored parameters                   */
    int n_memory;            /* m  - states in memory matrix D (Eq. 3)      */
    int n_remaining;         /* |L| - remaining training data (Eq. 4)       */

    mset_kernel_fn kernel;
    double kernel_param;
    double ridge;
    int normalize;

    double *lo, *span;       /* per-parameter min-max scaler (n)            */
    double *D_states;        /* m x n, rows = memory states (scaled)        */
    double *L_states;        /* |L| x n, remaining training data (scaled)   */
    int *D_index, *L_index;  /* indices into the original training set      */
    double *D_matrix;        /* n x m: columns are memory states (Eq. 3)    */
    double *G_inv;           /* m x m: (D^T (x) D)^-1  (Eq. 7)              */
    int fitted;
} MSET;

/* --- scaling ----------------------------------------------------------- */

static void mset_scale(const MSET *m, const double *state, double *out)
{
    if (!m->normalize) { memcpy(out, state, sizeof(double) * (size_t)m->n_params); return; }
    for (int j = 0; j < m->n_params; ++j)
        out[j] = (state[j] - m->lo[j]) / m->span[j];
}

static void mset_unscale(const MSET *m, const double *state, double *out)
{
    if (!m->normalize) { memcpy(out, state, sizeof(double) * (size_t)m->n_params); return; }
    for (int j = 0; j < m->n_params; ++j)
        out[j] = state[j] * m->span[j] + m->lo[j];
}

/* --- memory-matrix selection (paper: "three key procedures") ------------ */

typedef struct { double norm; int index; } NormIndex;

static int norm_index_cmp(const void *a, const void *b)
{
    double d = ((const NormIndex *)a)->norm - ((const NormIndex *)b)->norm;
    return (d > 0) - (d < 0);
}

/*
 * Select memory matrix D from scaled training data (l x n), exactly as the
 * paper describes:
 *   1. every state holding the min or max of any parameter (extreme features);
 *   2. order the remaining states by Euclidean norm;
 *   3. take additional states at equally spaced intervals from that ordering
 *      until D holds `memory_size` states;
 *   4. the rest is the remaining training data L (Eq. 4).
 *
 * `selected` must hold l ints; on return selected[i] is 1 if training state i
 * belongs to D. Returns the actual number of selected states.
 */
static int select_memory_matrix(const double *training, int l, int n,
                                int memory_size, int *selected)
{
    memset(selected, 0, sizeof(int) * (size_t)l);
    int count = 0;

    /* step 1: parameter extremes */
    for (int j = 0; j < n; ++j) {
        int imin = 0, imax = 0;
        for (int i = 1; i < l; ++i) {
            if (training[(size_t)i * n + j] < training[(size_t)imin * n + j]) imin = i;
            if (training[(size_t)i * n + j] > training[(size_t)imax * n + j]) imax = i;
        }
        if (!selected[imin]) { selected[imin] = 1; ++count; }
        if (!selected[imax]) { selected[imax] = 1; ++count; }
    }

    /* too many extremes: keep a norm-ordered spread of them */
    if (count > memory_size) {
        NormIndex *ext = malloc(sizeof(NormIndex) * (size_t)count);
        int k = 0;
        for (int i = 0; i < l; ++i)
            if (selected[i]) {
                ext[k].norm = euclidean_norm(&training[(size_t)i * n], n);
                ext[k].index = i;
                ++k;
            }
        qsort(ext, (size_t)count, sizeof(NormIndex), norm_index_cmp);
        memset(selected, 0, sizeof(int) * (size_t)l);
        double step = (memory_size > 1) ? (double)(count - 1) / (memory_size - 1) : 0.0;
        int kept = 0;
        for (int t = 0; t < memory_size; ++t) {
            int pos = (int)floor(t * step + 0.5);
            if (pos > count - 1) pos = count - 1;
            if (!selected[ext[pos].index]) { selected[ext[pos].index] = 1; ++kept; }
        }
        free(ext);
        return kept;
    }

    /* steps 2-3: fill from the norm-ordered remainder at equal intervals */
    int needed = memory_size - count;
    if (needed > 0) {
        NormIndex *rem = malloc(sizeof(NormIndex) * (size_t)l);
        int r = 0;
        for (int i = 0; i < l; ++i)
            if (!selected[i]) {
                rem[r].norm = euclidean_norm(&training[(size_t)i * n], n);
                rem[r].index = i;
                ++r;
            }
        qsort(rem, (size_t)r, sizeof(NormIndex), norm_index_cmp);

        if (r > 0) {
            if (needed == 1) {
                selected[rem[r / 2].index] = 1;
                ++count;
            } else {
                double step = (r > 1) ? (double)(r - 1) / (needed - 1) : 0.0;
                for (int t = 0; t < needed; ++t) {
                    int pos = (int)floor(t * step + 0.5);
                    if (pos > r - 1) pos = r - 1;
                    if (!selected[rem[pos].index]) { selected[rem[pos].index] = 1; ++count; }
                }
                /* backfill rounding duplicates */
                for (int i = 0; i < r && count < memory_size; ++i)
                    if (!selected[rem[i].index]) { selected[rem[i].index] = 1; ++count; }
            }
        }
        free(rem);
    }
    return count;
}

/* --- lifecycle ---------------------------------------------------------- */

static MSET *mset_create(mset_kernel_fn kernel, double kernel_param,
                         double ridge, int normalize)
{
    MSET *m = calloc(1, sizeof(MSET));
    if (!m) return NULL;
    m->kernel = kernel ? kernel : kernel_inverse_distance;
    m->kernel_param = (kernel_param > 0.0) ? kernel_param : 1.0;
    m->ridge = (ridge >= 0.0) ? ridge : 1e-8;
    m->normalize = normalize;
    return m;
}

static void mset_free(MSET *m)
{
    if (!m) return;
    free(m->lo); free(m->span);
    free(m->D_states); free(m->L_states);
    free(m->D_index); free(m->L_index);
    free(m->D_matrix); free(m->G_inv);
    free(m);
}

/*
 * Fit the model from healthy historic data (training: l states x n params,
 * row-major). Prerequisites stated in the paper (caller's responsibility):
 * the data must span every healthy operational state and contain no
 * anomalies, sensor failures or equipment failures.
 *
 * memory_size <= 0 selects the default of the Python version:
 *   min(l, max(2n, min(20, l/2)))  - rule of thumb of reference [6].
 *
 * Returns 0 on success, -1 on error.
 */
static int mset_fit(MSET *m, const double *training, int l, int n,
                    int memory_size)
{
    if (!m || !training || l < 2 || n < 1) return -1;
    m->n_params = n;

    /* min-max scaler */
    m->lo = malloc(sizeof(double) * (size_t)n);
    m->span = malloc(sizeof(double) * (size_t)n);
    if (!m->lo || !m->span) return -1;
    for (int j = 0; j < n; ++j) {
        double lo = training[j], hi = training[j];
        for (int i = 1; i < l; ++i) {
            double v = training[(size_t)i * n + j];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        m->lo[j] = lo;
        m->span[j] = (hi - lo > 1e-12) ? (hi - lo) : 1.0;
    }

    double *scaled = malloc(sizeof(double) * (size_t)l * n);
    if (!scaled) return -1;
    for (int i = 0; i < l; ++i)
        mset_scale(m, &training[(size_t)i * n], &scaled[(size_t)i * n]);

    if (memory_size <= 0) {
        int half = l / 2 > 0 ? l / 2 : 1;
        int cap = half < 20 ? half : 20;
        memory_size = 2 * n > cap ? 2 * n : cap;
        if (memory_size > l) memory_size = l;
    }
    if (memory_size > l) { free(scaled); return -1; }

    int *selected = malloc(sizeof(int) * (size_t)l);
    if (!selected) { free(scaled); return -1; }
    int mm = select_memory_matrix(scaled, l, n, memory_size, selected);

    m->n_memory = mm;
    m->n_remaining = l - mm;
    m->D_states = malloc(sizeof(double) * (size_t)mm * n);
    m->L_states = malloc(sizeof(double) * (size_t)m->n_remaining * n);
    m->D_index = malloc(sizeof(int) * (size_t)mm);
    m->L_index = malloc(sizeof(int) * (size_t)(m->n_remaining > 0 ? m->n_remaining : 1));
    m->D_matrix = malloc(sizeof(double) * (size_t)n * mm);
    m->G_inv = malloc(sizeof(double) * (size_t)mm * mm);
    if (!m->D_states || !m->L_states || !m->D_index || !m->L_index ||
        !m->D_matrix || !m->G_inv) { free(scaled); free(selected); return -1; }

    int di = 0, li = 0;
    for (int i = 0; i < l; ++i) {
        if (selected[i]) {
            memcpy(&m->D_states[(size_t)di * n], &scaled[(size_t)i * n],
                   sizeof(double) * (size_t)n);
            m->D_index[di++] = i;
        } else {
            memcpy(&m->L_states[(size_t)li * n], &scaled[(size_t)i * n],
                   sizeof(double) * (size_t)n);
            m->L_index[li++] = i;
        }
    }
    free(selected);
    free(scaled);

    /* D as n x m matrix whose columns are memory states (Eq. 3) */
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < mm; ++i)
            m->D_matrix[(size_t)j * mm + i] = m->D_states[(size_t)i * n + j];

    /* G = D^T (x) D (m x m similarity matrix), inverted once (Eq. 7) */
    double *g = malloc(sizeof(double) * (size_t)mm * mm);
    if (!g) return -1;
    for (int i = 0; i < mm; ++i)
        for (int j = i; j < mm; ++j) {
            double s = m->kernel(&m->D_states[(size_t)i * n],
                                 &m->D_states[(size_t)j * n], n, m->kernel_param);
            g[(size_t)i * mm + j] = s;
            g[(size_t)j * mm + i] = s;
        }
    int rc = invert(g, m->G_inv, mm, m->ridge);
    free(g);
    if (rc != 0) {
        fprintf(stderr, "mset_fit: similarity matrix singular; "
                        "increase ridge or reduce memory_size\n");
        return -1;
    }
    m->fitted = 1;
    return 0;
}

/* --- estimation --------------------------------------------------------- */

/* Eq. (7): W = (D^T (x) D)^-1 . (D^T (x) Xobs).  w must hold n_memory. */
static int mset_weights(const MSET *m, const double *observation, double *w)
{
    if (!m->fitted) return -1;
    int n = m->n_params, mm = m->n_memory;
    double *x = malloc(sizeof(double) * (size_t)n);
    double *sim = malloc(sizeof(double) * (size_t)mm);
    if (!x || !sim) { free(x); free(sim); return -1; }
    mset_scale(m, observation, x);
    for (int i = 0; i < mm; ++i)
        sim[i] = m->kernel(&m->D_states[(size_t)i * n], x, n, m->kernel_param);
    matvec(m->G_inv, sim, w, mm, mm);
    free(x); free(sim);
    return 0;
}

/* Eq. (5)/(8): Xest = D . W, returned in the original units. est holds n. */
MSET_API int mset_estimate(const MSET *m, const double *observation, double *est)
{
    int n = m->n_params, mm = m->n_memory;
    double *w = malloc(sizeof(double) * (size_t)mm);
    double *es = malloc(sizeof(double) * (size_t)n);
    if (!w || !es) { free(w); free(es); return -1; }
    int rc = mset_weights(m, observation, w);
    if (rc == 0) {
        matvec(m->D_matrix, w, es, n, mm);
        mset_unscale(m, es, est);
    }
    free(w); free(es);
    return rc;
}

/*
 * Actual residual R = Xest - Xobs (Fig. 2). res holds n.
 * scaled != 0 returns the residual in normalised parameter units - what makes
 * the Euclidean norm of Eq. (10) meaningful across degC / %RH / g.
 */
static int mset_residual(const MSET *m, const double *observation, double *res,
                         int scaled)
{
    int n = m->n_params, mm = m->n_memory;
    double *w = malloc(sizeof(double) * (size_t)mm);
    double *x = malloc(sizeof(double) * (size_t)n);
    double *es = malloc(sizeof(double) * (size_t)n);
    if (!w || !x || !es) { free(w); free(x); free(es); return -1; }

    int rc = mset_weights(m, observation, w);
    if (rc == 0) {
        matvec(m->D_matrix, w, es, n, mm);              /* scaled estimate  */
        if (scaled) {
            mset_scale(m, observation, x);
            for (int j = 0; j < n; ++j) res[j] = es[j] - x[j];
        } else {
            mset_unscale(m, es, x);                     /* unscaled estimate */
            for (int j = 0; j < n; ++j) res[j] = x[j] - observation[j];
        }
    }
    free(w); free(x); free(es);
    return rc;
}

/* Residuals of a whole series (k x n each, row-major). */
static int mset_residuals(const MSET *m, const double *observations, int k,
                          double *residuals, int scaled)
{
    for (int i = 0; i < k; ++i)
        if (mset_residual(m, &observations[(size_t)i * m->n_params],
                          &residuals[(size_t)i * m->n_params], scaled) != 0)
            return -1;
    return 0;
}

/*
 * Fig. 2: healthy residuals RL = Lest - L. The remaining training data L is
 * healthy by construction, so these residuals characterise the healthy state
 * and form the reference distribution for fault detection.
 * out must hold n_remaining x n_params.
 */
static int mset_healthy_residuals(const MSET *m, double *out)
{
    int n = m->n_params;
    double *raw = malloc(sizeof(double) * (size_t)n);
    if (!raw) return -1;
    for (int i = 0; i < m->n_remaining; ++i) {
        mset_unscale(m, &m->L_states[(size_t)i * n], raw);
        if (mset_residual(m, raw, &out[(size_t)i * n], 1) != 0) {
            free(raw);
            return -1;
        }
    }
    free(raw);
    return 0;
}

/* Per-parameter (mean, stddev) of the healthy residuals. mu, sd hold n. */
static int mset_healthy_statistics(const MSET *m, double *mu, double *sd)
{
    int n = m->n_params, k = m->n_remaining;
    if (k < 1) {
        for (int j = 0; j < n; ++j) { mu[j] = 0.0; sd[j] = 1e-6; }
        return 0;
    }
    double *rl = malloc(sizeof(double) * (size_t)k * n);
    if (!rl) return -1;
    if (mset_healthy_residuals(m, rl) != 0) { free(rl); return -1; }
    for (int j = 0; j < n; ++j) {
        double s = 0.0;
        for (int i = 0; i < k; ++i) s += rl[(size_t)i * n + j];
        mu[j] = s / k;
        double var = 0.0;
        if (k > 1) {
            for (int i = 0; i < k; ++i) {
                double d = rl[(size_t)i * n + j] - mu[j];
                var += d * d;
            }
            var /= (k - 1);
        }
        sd[j] = sqrt(var);
        if (sd[j] < 1e-9) sd[j] = 1e-9;
    }
    free(rl);
    return 0;
}

/* ======================================================================== */
/* 4. Fault detection - two-sided SPRT (Fig. 2)                             */
/* ======================================================================== */

typedef enum { SPRT_CONTINUE = 0, SPRT_HEALTHY = 1, SPRT_FAULT = 2 } SprtVerdict;

/*
 * Wald's Sequential Probability Ratio Test for a mean shift in the residual
 * stream. Two hypotheses tracked simultaneously (positive and negative shift):
 * an MSET residual R = Xest - Xobs drifts negative when the parameter rises
 * above its healthy estimate and positive when it falls below, so a one-sided
 * test would miss half of all degradation modes.
 *
 *     H0 : r ~ N(mu0, sigma^2)               (healthy)
 *     H1+: r ~ N(mu0 + M*sigma, sigma^2)     (positive drift)
 *     H1-: r ~ N(mu0 - M*sigma, sigma^2)     (negative drift)
 *
 * Boundaries:  ln(beta/(1-alpha)) < SUM ln(LR) < ln((1-beta)/alpha)
 */
typedef struct {
    double mu0, sigma, shift;
    double upper, lower;
    double llr_pos, llr_neg;
} SPRT;

static void sprt_init(SPRT *s, double mean, double sigma,
                      double alpha, double beta, double disturbance)
{
    s->mu0 = mean;
    s->sigma = (sigma > 1e-12) ? sigma : 1e-12;
    s->shift = disturbance * s->sigma;
    s->upper = log((1.0 - beta) / alpha);
    s->lower = log(beta / (1.0 - alpha));
    s->llr_pos = s->llr_neg = 0.0;
}

/* Feed one residual sample. */
static SprtVerdict sprt_update(SPRT *s, double value)
{
    double gain = s->shift / (s->sigma * s->sigma);
    double dev = value - s->mu0;
    s->llr_pos += gain * (dev - s->shift / 2.0);
    s->llr_neg += gain * (-dev - s->shift / 2.0);

    if (s->llr_pos >= s->upper || s->llr_neg >= s->upper) {
        s->llr_pos = s->llr_neg = 0.0;
        return SPRT_FAULT;
    }
    if (s->llr_pos <= s->lower && s->llr_neg <= s->lower) {
        s->llr_pos = s->llr_neg = 0.0;
        return SPRT_HEALTHY;
    }
    if (s->llr_pos < s->lower) s->llr_pos = s->lower;
    if (s->llr_neg < s->lower) s->llr_neg = s->lower;
    return SPRT_CONTINUE;
}

/* First fault index (0-based) in a series, or -1 if none. */
static int sprt_first_fault(SPRT *s, const double *series, int k)
{
    for (int i = 0; i < k; ++i)
        if (sprt_update(s, series[i]) == SPRT_FAULT) return i;
    return -1;
}

/* ======================================================================== */
/* 5. Degradation model (Eq. 9, 10, 11)                                     */
/* ======================================================================== */

/* Eq. (9): De_Ac(tk) = SUM_{i=1..k} ||R(ti)||. out holds k running values.  */
static void accumulated_degradation(const double *residuals, int k, int n,
                                    double *out)
{
    double total = 0.0;
    for (int i = 0; i < k; ++i) {
        total += euclidean_norm(&residuals[(size_t)i * n], n);
        out[i] = total;
    }
}

/*
 * Eq. (11): De(tk) = (1/k) SUM_{i=1..k} ||R(ti)|| - the sample-count-
 * normalised degradation that removes Eq. (9)'s dependence on how many
 * samples happen to have been taken. This is the quantity of Figs. 4-6 and
 * the one used for RUL prediction. out holds k values.
 */
static void degradation(const double *residuals, int k, int n, double *out)
{
    accumulated_degradation(residuals, k, n, out);
    for (int i = 0; i < k; ++i) out[i] /= (i + 1);
}

/* ======================================================================== */
/* 6. Criteria of failure (Fig. 3)                                          */
/* ======================================================================== */

/*
 * The degradation at which the product is declared failed, estimated
 * statistically from units that already failed (historic failures with a
 * similar PoF, or an accelerated test). Paper's assumptions: the same
 * products fail at the same degradation if their PoF is similar, and the
 * degradation at failure is unchanged by acceleration even though the
 * time-to-failure is not. The paper adopts mean +/- k_sigma*sigma as the band.
 */
typedef struct {
    double mean, sigma, k_sigma;
    double lower, upper;
} FailureCriteria;

static int failure_criteria_init(FailureCriteria *c,
                                 const double *failure_degradations, int count,
                                 double k_sigma)
{
    if (count < 2) return -1;   /* need at least two failed units */
    double s = 0.0;
    for (int i = 0; i < count; ++i) s += failure_degradations[i];
    c->mean = s / count;
    double var = 0.0;
    for (int i = 0; i < count; ++i) {
        double d = failure_degradations[i] - c->mean;
        var += d * d;
    }
    c->sigma = sqrt(var / (count - 1));
    c->k_sigma = k_sigma;
    c->lower = c->mean - k_sigma * c->sigma;
    c->upper = c->mean + k_sigma * c->sigma;
    return 0;
}

/* ======================================================================== */
/* 7. Degradation regression and RUL prediction (Fig. 3, Figs. 5-6)         */
/* ======================================================================== */

/*
 * Ordinary least-squares straight line De(t) = intercept + slope*t.
 * The paper deliberately uses linear regression "to make it easy to figure
 * out equations"; power_trend_fit below is the non-linear alternative.
 */
typedef struct {
    double slope, intercept, r_squared;
} LinearTrend;

static int linear_trend_fit(LinearTrend *t, const double *times,
                            const double *values, int k)
{
    if (k < 2) return -1;
    double mt = 0.0, mv = 0.0;
    for (int i = 0; i < k; ++i) { mt += times[i]; mv += values[i]; }
    mt /= k; mv /= k;
    double sxx = 0.0, sxy = 0.0;
    for (int i = 0; i < k; ++i) {
        sxx += (times[i] - mt) * (times[i] - mt);
        sxy += (times[i] - mt) * (values[i] - mv);
    }
    if (sxx <= 0.0) return -1;
    t->slope = sxy / sxx;
    t->intercept = mv - t->slope * mt;

    double ss_tot = 0.0, ss_res = 0.0;
    for (int i = 0; i < k; ++i) {
        double p = t->intercept + t->slope * times[i];
        ss_tot += (values[i] - mv) * (values[i] - mv);
        ss_res += (values[i] - p) * (values[i] - p);
    }
    t->r_squared = (ss_tot > 0.0) ? 1.0 - ss_res / ss_tot : 1.0;
    return 0;
}

static double linear_trend_predict(const LinearTrend *t, double time)
{
    return t->intercept + t->slope * time;
}

/* Solve De(t) = level. Returns NAN if the trend never reaches `level`. */
static double linear_trend_time_to_reach(const LinearTrend *t, double level)
{
    if (t->slope <= 0.0) return NAN;
    return (level - t->intercept) / t->slope;
}

/* Log-log least squares: De(t) = a * t^b  (t > 0). */
typedef struct {
    double a, b, r_squared;
} PowerTrend;

MSET_API int power_trend_fit(PowerTrend *p, const double *times,
                           const double *values, int k)
{
    double *lt = malloc(sizeof(double) * (size_t)k);
    double *lv = malloc(sizeof(double) * (size_t)k);
    if (!lt || !lv) { free(lt); free(lv); return -1; }
    int c = 0;
    for (int i = 0; i < k; ++i)
        if (times[i] > 0.0 && values[i] > 0.0) {
            lt[c] = log(times[i]);
            lv[c] = log(values[i]);
            ++c;
        }
    LinearTrend fit;
    int rc = linear_trend_fit(&fit, lt, lv, c);
    free(lt); free(lv);
    if (rc != 0) return -1;
    p->b = fit.slope;
    p->a = exp(fit.intercept);
    p->r_squared = fit.r_squared;
    return 0;
}

MSET_API double power_trend_time_to_reach(const PowerTrend *p, double level)
{
    if (level <= 0.0 || p->a <= 0.0 || p->b <= 0.0) return NAN;
    return exp((log(level) - log(p->a)) / p->b);
}

/* Result container: failure-time interval and remaining useful life.
 * NAN in any field means "the trend never reaches that level". */
typedef struct {
    double current_time;
    double failure_time_lower, failure_time_nominal, failure_time_upper;
    double rul_lower, rul_nominal, rul_upper;
    LinearTrend trend;
    FailureCriteria criteria;
} RULPrediction;

static double rul_from(double failure_time, double now)
{
    if (isnan(failure_time)) return NAN;
    double r = failure_time - now;
    return r > 0.0 ? r : 0.0;
}

/*
 * The RUL prediction of Figure 3 / Figure 6: regress the observed degradation
 * trend, extrapolate, and read off the times at which it crosses the lower,
 * mean and upper failure criteria. RUL = crossing time - current time.
 */
static int predict_rul(RULPrediction *out, const double *times,
                       const double *degradation_series, int k,
                       const FailureCriteria *criteria)
{
    if (linear_trend_fit(&out->trend, times, degradation_series, k) != 0)
        return -1;
    out->criteria = *criteria;
    out->current_time = times[k - 1];
    out->failure_time_lower = linear_trend_time_to_reach(&out->trend, criteria->lower);
    out->failure_time_nominal = linear_trend_time_to_reach(&out->trend, criteria->mean);
    out->failure_time_upper = linear_trend_time_to_reach(&out->trend, criteria->upper);
    out->rul_lower = rul_from(out->failure_time_lower, out->current_time);
    out->rul_nominal = rul_from(out->failure_time_nominal, out->current_time);
    out->rul_upper = rul_from(out->failure_time_upper, out->current_time);
    return 0;
}

/* ======================================================================== */
/* 8. Console helpers                                                       */
/* ======================================================================== */

static void fmt_optional(double v, char *buf, size_t size)
{
    if (isnan(v)) snprintf(buf, size, "never");
    else          snprintf(buf, size, "%.0f", v);
}

#define PLOT_W 62
#define PLOT_H 16

/*
 * Tiny ASCII scatter plot (same as the Python version) so results are visible
 * without a plotting library. series: n_series x k values, markers: one char
 * per series, times: k values.
 */
static void ascii_plot(const char *title, const double *times,
                       const double *series, const char *markers,
                       int n_series, int k, int height)
{
    if (height <= 0 || height > PLOT_H) height = PLOT_H;
    double vmin = series[0], vmax = series[0];
    for (int s = 0; s < n_series; ++s)
        for (int i = 0; i < k; ++i) {
            double v = series[(size_t)s * k + i];
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
    if (vmax - vmin < 1e-12) vmax = vmin + 1.0;
    double tmin = times[0], tmax = times[k - 1];
    for (int i = 0; i < k; ++i) {
        if (times[i] < tmin) tmin = times[i];
        if (times[i] > tmax) tmax = times[i];
    }
    if (tmax - tmin < 1e-12) tmax = tmin + 1.0;

    char grid[PLOT_H][PLOT_W];
    memset(grid, ' ', sizeof(grid));
    for (int s = 0; s < n_series; ++s)
        for (int i = 0; i < k; ++i) {
            int col = (int)((times[i] - tmin) / (tmax - tmin) * (PLOT_W - 1) + 0.5);
            double v = series[(size_t)s * k + i];
            int row = height - 1 - (int)((v - vmin) / (vmax - vmin) * (height - 1) + 0.5);
            if (col >= 0 && col < PLOT_W && row >= 0 && row < height)
                grid[row][col] = markers[s];
        }

    if (title && *title) printf("%s\n", title);
    for (int r = 0; r < height; ++r) {
        if (r == 0)               printf("%7.3f |", vmax);
        else if (r == height - 1) printf("%7.3f |", vmin);
        else                      printf("        |");
        fwrite(grid[r], 1, PLOT_W, stdout);
        putchar('\n');
    }
    printf("        +");
    for (int i = 0; i < PLOT_W; ++i) putchar('-');
    printf("\n         %-10g%*s%g\n", tmin, PLOT_W - 20, "", tmax);
}

/* ======================================================================== */
/* 9. Deterministic PRNG + synthetic data generator for the demonstration   */
/* ======================================================================== */

/* xorshift64* PRNG - seedable and reproducible across platforms. */
typedef struct { unsigned long long s; int has_gauss; double gauss; } Rng;

static void rng_init(Rng *r, unsigned long long seed)
{
    r->s = seed ? seed : 0x9E3779B97F4A7C15ULL;
    r->has_gauss = 0;
    r->gauss = 0.0;
}

static double rng_uniform(Rng *r)   /* (0, 1) */
{
    r->s ^= r->s >> 12;
    r->s ^= r->s << 25;
    r->s ^= r->s >> 27;
    unsigned long long x = r->s * 0x2545F4914F6CDD1DULL;
    return ((double)(x >> 11) + 1.0) / 9007199254740994.0;
}

static double rng_gauss(Rng *r, double mu, double sigma)  /* Box-Muller */
{
    if (r->has_gauss) {
        r->has_gauss = 0;
        return mu + sigma * r->gauss;
    }
    double u1 = rng_uniform(r), u2 = rng_uniform(r);
    double radius = sqrt(-2.0 * log(u1)), theta = 2.0 * M_PI * u2;
    r->gauss = radius * sin(theta);
    r->has_gauss = 1;
    return mu + sigma * radius * cos(theta);
}

#define DEMO_PARAMS 3

/*
 * Three monitored parameters of an electronic component, one sample per day:
 *   p1 temperature [degC], p2 relative humidity [%], p3 vibration [g].
 * A healthy unit tracks its nominal envelope; a degrading unit develops a
 * slow multi-parameter drift that MSET turns into a growing residual.
 * out must hold days x DEMO_PARAMS.
 */
static void generate_unit(double *out, int days, double drift_rate, Rng *rng,
                          double noise)
{
    for (int day = 0; day < days; ++day) {
        double cycle = sin(2.0 * M_PI * day / 7.0);   /* weekly duty cycle */
        double drift = drift_rate * day;
        out[(size_t)day * DEMO_PARAMS + 0] =
            55.0 + 4.0 * cycle + rng_gauss(rng, 0.0, noise) + 6.0 * drift;
        out[(size_t)day * DEMO_PARAMS + 1] =
            45.0 + 6.0 * cycle + rng_gauss(rng, 0.0, noise * 2.0) + 9.0 * drift;
        out[(size_t)day * DEMO_PARAMS + 2] =
            0.80 + 0.10 * cycle + rng_gauss(rng, 0.0, noise * 0.02) + 0.25 * drift;
    }
}

/* ======================================================================== */
/* 10. Demo - full worked example (structure of the paper's case study)     */
/* ======================================================================== */

static void demo(void)
{
    Rng rng;
    rng_init(&rng, 20070101ULL);

    printf("==========================================================================\n");
    printf("MSET-based RUL prediction - worked example (C implementation)\n");
    printf("==========================================================================\n");

    /* ---- healthy history ------------------------------------------- */
    enum { TRAIN_DAYS = 60 };
    double *training = malloc(sizeof(double) * TRAIN_DAYS * DEMO_PARAMS);
    generate_unit(training, TRAIN_DAYS, 0.0, &rng, 0.35);

    MSET *model = mset_create(kernel_inverse_distance, 0.30, 1e-8, 1);
    if (mset_fit(model, training, TRAIN_DAYS, DEMO_PARAMS, 14) != 0) {
        fprintf(stderr, "demo: mset_fit failed\n");
        goto cleanup_training;
    }
    printf("\nTraining states l = %d, parameters n = %d\n", TRAIN_DAYS, model->n_params);
    printf("Memory matrix D  m = %d  (rule of thumb: m >= 2n = %d)\n",
           model->n_memory, 2 * model->n_params);
    printf("Remaining training data L = %d states\n", model->n_remaining);

    double mu[DEMO_PARAMS], sd[DEMO_PARAMS];
    mset_healthy_statistics(model, mu, sd);
    for (int j = 0; j < DEMO_PARAMS; ++j)
        printf("  healthy residual param %d: mean %+.5f, sigma %.5f\n",
               j + 1, mu[j], sd[j]);

    /* ---- accelerated test: six units, five fail --------------------- */
    printf("\n--- Accelerated test (criteria of failure) ---\n");
    enum { ACC_DAYS = 28, N_FAILED = 5 };
    const double rates[N_FAILED] = { 0.030, 0.033, 0.062, 0.040, 0.070 };
    const int failure_days[N_FAILED] = { 22, 22, 11, 23, 10 };

    double *acc_units = malloc(sizeof(double) * N_FAILED * ACC_DAYS * DEMO_PARAMS);
    double *acc_res = malloc(sizeof(double) * ACC_DAYS * DEMO_PARAMS);
    double *acc_de = malloc(sizeof(double) * N_FAILED * ACC_DAYS);
    double fail_degradations[N_FAILED];

    printf("%-8s%20s%14s\n", "Unit", "Failure time (day)", "Degradation");
    for (int u = 0; u < N_FAILED; ++u) {
        double *unit = &acc_units[(size_t)u * ACC_DAYS * DEMO_PARAMS];
        generate_unit(unit, ACC_DAYS, rates[u], &rng, 0.35);
        mset_residuals(model, unit, ACC_DAYS, acc_res, 1);
        degradation(acc_res, ACC_DAYS, DEMO_PARAMS, &acc_de[(size_t)u * ACC_DAYS]);
        fail_degradations[u] = acc_de[(size_t)u * ACC_DAYS + failure_days[u] - 1];
        printf("Test %-3d%20d%14.3f\n", u + 1, failure_days[u], fail_degradations[u]);
    }

    double *healthy_unit = malloc(sizeof(double) * ACC_DAYS * DEMO_PARAMS);
    double healthy_de[ACC_DAYS];
    generate_unit(healthy_unit, ACC_DAYS, 0.0005, &rng, 0.35);
    mset_residuals(model, healthy_unit, ACC_DAYS, acc_res, 1);
    degradation(acc_res, ACC_DAYS, DEMO_PARAMS, healthy_de);
    printf("%-8s%20s%14.3f\n", "Test 6", "(healthy)", healthy_de[ACC_DAYS - 1]);

    FailureCriteria criteria;
    failure_criteria_init(&criteria, fail_degradations, N_FAILED, 1.0);
    printf("\nMean degradation at failure : %.3f\n", criteria.mean);
    printf("Standard deviation (sigma)  : %.3f\n", criteria.sigma);
    printf("Criteria of failure         : [%.3f, %.3f]\n",
           criteria.lower, criteria.upper);

    {
        double acc_times[ACC_DAYS];
        for (int i = 0; i < ACC_DAYS; ++i) acc_times[i] = i + 1;
        putchar('\n');
        ascii_plot("Fig. 4 analogue - degradation of the accelerated-test units",
                   acc_times, acc_de, "12345", N_FAILED, ACC_DAYS, PLOT_H);
    }

    /* ---- the monitored ("7th") unit --------------------------------- */
    printf("\n--- Monitored unit ---\n");
    enum { MON_DAYS = 24 };
    double *test_unit = malloc(sizeof(double) * MON_DAYS * DEMO_PARAMS);
    double *test_res = malloc(sizeof(double) * MON_DAYS * DEMO_PARAMS);
    double test_de[MON_DAYS], times[MON_DAYS];
    generate_unit(test_unit, MON_DAYS, 0.014, &rng, 0.35);
    for (int i = 0; i < MON_DAYS; ++i) times[i] = i + 1;
    mset_residuals(model, test_unit, MON_DAYS, test_res, 1);
    degradation(test_res, MON_DAYS, DEMO_PARAMS, test_de);

    /* SPRT fault detection per parameter (Fig. 2 block) */
    printf("SPRT fault alerts (monitored unit), first alert day per parameter: ");
    for (int j = 0; j < DEMO_PARAMS; ++j) {
        double channel[MON_DAYS];
        for (int i = 0; i < MON_DAYS; ++i)
            channel[i] = test_res[(size_t)i * DEMO_PARAMS + j];
        SPRT sprt;
        sprt_init(&sprt, mu[j], sd[j], 0.01, 0.01, 3.0);
        int first = sprt_first_fault(&sprt, channel, MON_DAYS);
        if (first >= 0) printf("p%d=%d%s", j + 1, first + 1,
                               j < DEMO_PARAMS - 1 ? ", " : "\n");
        else            printf("p%d=none%s", j + 1,
                               j < DEMO_PARAMS - 1 ? ", " : "\n");
    }

    RULPrediction pred;
    if (predict_rul(&pred, times, test_de, MON_DAYS, &criteria) == 0) {
        char lo[16], hi[16], nom[16];
        printf("\nDegradation trend           : De = %.4f + %.6f*t  (R^2=%.4f)\n",
               pred.trend.intercept, pred.trend.slope, pred.trend.r_squared);
        printf("Current degradation (t=%g)  : %.4f\n",
               times[MON_DAYS - 1], test_de[MON_DAYS - 1]);
        fmt_optional(pred.failure_time_lower, lo, sizeof lo);
        fmt_optional(pred.failure_time_upper, hi, sizeof hi);
        fmt_optional(pred.failure_time_nominal, nom, sizeof nom);
        printf("Predicted failure time      : [%s, %s] days (nominal %s)\n", lo, hi, nom);
        fmt_optional(pred.rul_lower, lo, sizeof lo);
        fmt_optional(pred.rul_upper, hi, sizeof hi);
        fmt_optional(pred.rul_nominal, nom, sizeof nom);
        printf("Remaining useful life       : [%s, %s] days (nominal %s)\n", lo, hi, nom);
    }

    putchar('\n');
    ascii_plot("Fig. 5 analogue - degradation of the monitored unit",
               times, test_de, "o", 1, MON_DAYS, 12);

    free(test_unit); free(test_res);
    free(healthy_unit);
    free(acc_units); free(acc_res); free(acc_de);
cleanup_training:
    mset_free(model);
    free(training);
}

/* ======================================================================== */
/* 11. Numerical reproduction of the paper's published case study           */
/* ======================================================================== */

static void reproduce_paper_case_study(void)
{
    printf("\n==========================================================================\n");
    printf("Reproduction of the paper's published case study\n");
    printf("==========================================================================\n");

    /* Table 1: degradations at failure -> mean 1.9, sigma 0.47, [1.43, 2.37] */
    const double table_1[] = { 1.46, 1.63, 1.94, 2.68, 1.77 };
    FailureCriteria criteria;
    failure_criteria_init(&criteria, table_1, 5, 1.0);
    printf("\nTable 1 degradations        : 1.46 1.63 1.94 2.68 1.77\n");
    printf("Mean   computed / published : %.2f / 1.90\n", criteria.mean);
    printf("Sigma  computed / published : %.2f / 0.47\n", criteria.sigma);
    printf("Criteria computed / published: [%.2f, %.2f] / [1.43, 2.37]\n",
           criteria.lower, criteria.upper);

    /* Figure 5: 18 daily degradation readings rising ~0.165 -> ~0.29,
     * published prediction: failure time in [154, 268] days. */
    enum { K = 18 };
    double times[K];
    const double fig5[K] = {
        0.165, 0.197, 0.172, 0.163, 0.157, 0.168, 0.222, 0.227,
        0.232, 0.228, 0.238, 0.248, 0.258, 0.283, 0.272, 0.277,
        0.293, 0.284
    };
    for (int i = 0; i < K; ++i) times[i] = i + 1;

    RULPrediction pred;
    if (predict_rul(&pred, times, fig5, K, &criteria) == 0) {
        char lo[16], hi[16], nom[16];
        printf("\nFigure 5 regression         : De = %.4f + %.6f*t  (R^2=%.4f)\n",
               pred.trend.intercept, pred.trend.slope, pred.trend.r_squared);
        fmt_optional(pred.failure_time_lower, lo, sizeof lo);
        fmt_optional(pred.failure_time_upper, hi, sizeof hi);
        fmt_optional(pred.failure_time_nominal, nom, sizeof nom);
        printf("Failure time computed       : [%s, %s] days (nominal %s)\n", lo, hi, nom);
        printf("Failure time published      : [154, 268] days (nominal 210)\n");
        fmt_optional(pred.rul_lower, lo, sizeof lo);
        fmt_optional(pred.rul_upper, hi, sizeof hi);
        printf("RUL at day %g               : [%s, %s] days\n",
               times[K - 1], lo, hi);
    }

    putchar('\n');
    ascii_plot("Figure 5 - degradation of the test component",
               times, fig5, "*", 1, K, 10);

    /* Extrapolated view corresponding to Figure 6 */
    enum { H = 31 };
    double horizon[H], lines[3 * H];
    for (int i = 0; i < H; ++i) {
        horizon[i] = 10.0 * i;
        lines[0 * H + i] = criteria.lower;
        lines[1 * H + i] = criteria.upper;
        lines[2 * H + i] = linear_trend_predict(&pred.trend, horizon[i]);
    }
    putchar('\n');
    ascii_plot("Figure 6 - projected degradation ( / ) vs failure criteria "
               "( - lower, = upper )",
               horizon, lines, "-=/", 3, H, 14);
}

/* ======================================================================== */

int main(void)
{
    demo();
    reproduce_paper_case_study();
    printf("\nDone.\n");
    return 0;
}
