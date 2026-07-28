/*
 * csv_io.c
 * ========
 *
 * See csv_io.h for the public API and format contract. Implementation notes:
 *
 *   - read_line() grows a buffer with fgets()+realloc() rather than using
 *     POSIX getline(), so this stays plain C99 with no platform-specific
 *     feature-test macros (consistent with the rest of this codebase).
 *   - split_fields() is a small hand-written tokenizer: splits on `delimiter`
 *     outside of a simple double-quoted span, trims surrounding whitespace
 *     per field. It does not support RFC4180 escaped-quote-doubling, which
 *     is not needed for numeric data.
 *   - Column count is fixed by the first data row; every later row must
 *     match exactly, or csv_read() fails with a line-numbered diagnostic -
 *     required for MSET's rectangular matrix math, and it's how "adapts to
 *     any size matrix" stays honest rather than silently padding/truncating
 *     a malformed file.
 */

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "csv_io.h"

/* ======================================================================== */
/* Line reading                                                             */
/* ======================================================================== */

/*
 * Read one line of arbitrary length from `f` into a freshly malloc'd,
 * NUL-terminated buffer with any trailing \n / \r\n stripped. Returns NULL
 * at true end-of-file (nothing left to read); caller frees the result.
 */
static char *read_line(FILE *f)
{
    size_t cap = 256, len = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    buf[0] = '\0';

    int got_any = 0;
    for (;;) {
        if (len + 1 >= cap) {
            size_t new_cap = cap * 2;
            char *grown = realloc(buf, new_cap);
            if (!grown) { free(buf); return NULL; }
            buf = grown;
            cap = new_cap;
        }
        if (!fgets(buf + len, (int)(cap - len), f)) {
            if (!got_any) { free(buf); return NULL; }  /* true EOF */
            break;                                      /* EOF after partial line */
        }
        got_any = 1;
        size_t chunk_len = strlen(buf + len);
        len += chunk_len;
        if (len > 0 && buf[len - 1] == '\n') {
            --len;
            if (len > 0 && buf[len - 1] == '\r') --len;
            buf[len] = '\0';
            break;
        }
        /* fgets filled the buffer without hitting '\n' - grow and continue */
    }
    return buf;
}

static int is_blank_line(const char *s)
{
    for (const char *p = s; *p; ++p)
        if (*p != ' ' && *p != '\t' && *p != '\r' && *p != '\n')
            return 0;
    return 1;
}

/* ======================================================================== */
/* Field splitting                                                          */
/* ======================================================================== */

/*
 * Split `line` on `delimiter` (outside simple double-quoted spans),
 * trimming leading/trailing spaces and tabs from each field. Returns a
 * malloc'd array of malloc'd, NUL-terminated field strings and sets
 * *out_count; caller frees each string and the array. On allocation
 * failure, frees everything itself and sets *out_count = -1.
 */
static char **split_fields(const char *line, char delimiter, int *out_count)
{
    size_t len = strlen(line);
    size_t cap = 8;
    char **fields = malloc(sizeof(char *) * cap);
    int n = 0;
    size_t i = 0;

    if (!fields) { *out_count = -1; return NULL; }

    for (;;) {
        while (i < len && (line[i] == ' ' || line[i] == '\t')) ++i;

        char *field;
        if (i < len && line[i] == '"') {
            ++i;
            size_t start = i;
            while (i < len && line[i] != '"') ++i;
            size_t flen = i - start;
            field = malloc(flen + 1);
            if (!field) goto oom;
            memcpy(field, line + start, flen);
            field[flen] = '\0';
            if (i < len && line[i] == '"') ++i;
            while (i < len && (line[i] == ' ' || line[i] == '\t')) ++i;
            if (i < len && line[i] == delimiter) ++i;
        } else {
            size_t start = i;
            while (i < len && line[i] != delimiter) ++i;
            size_t end = i;
            while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\t')) --end;
            size_t flen = end - start;
            field = malloc(flen + 1);
            if (!field) goto oom;
            memcpy(field, line + start, flen);
            field[flen] = '\0';
            if (i < len && line[i] == delimiter) ++i;
        }

        if ((size_t)n == cap) {
            cap *= 2;
            char **grown = realloc(fields, sizeof(char *) * cap);
            if (!grown) { free(field); goto oom; }
            fields = grown;
        }
        fields[n++] = field;

        if (i >= len) break;
    }

    /* Drop one trailing empty field from a trailing delimiter (a common
     * spreadsheet-export artifact), but never the only field on the line. */
    if (n > 1 && fields[n - 1][0] == '\0') {
        free(fields[n - 1]);
        --n;
    }

    *out_count = n;
    return fields;

oom:
    for (int k = 0; k < n; ++k) free(fields[k]);
    free(fields);
    *out_count = -1;
    return NULL;
}

static void free_fields(char **fields, int count)
{
    if (!fields) return;
    for (int j = 0; j < count; ++j) free(fields[j]);
    free(fields);
}

/* ======================================================================== */
/* Number parsing                                                           */
/* ======================================================================== */

static int parse_double(const char *field, double *out)
{
    while (*field == ' ' || *field == '\t') ++field;
    if (*field == '\0') return -1;               /* empty field */
    char *end = NULL;
    double v = strtod(field, &end);
    if (end == field) return -1;                  /* no characters consumed */
    while (*end == ' ' || *end == '\t') ++end;
    if (*end != '\0') return -1;                   /* trailing garbage */
    *out = v;
    return 0;
}

/* ======================================================================== */
/* csv_read / csv_read_matrix                                               */
/* ======================================================================== */

int csv_read(const char *path, int has_header, char delimiter, CsvMatrix *out)
{
    if (!path || !out) return -1;

    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "csv_read: cannot open '%s': %s\n", path, strerror(errno));
        return -1;
    }

    char **headers = NULL;
    int header_cols = 0;
    int have_header = 0;

    double *data = NULL;
    size_t row_cap = 0;
    int rows = 0;
    int cols = -1;

    int line_no = 0;
    int rc = 0;
    char *line;

    while ((line = read_line(f)) != NULL) {
        ++line_no;
        if (is_blank_line(line)) { free(line); continue; }

        int field_count = 0;
        char **fields = split_fields(line, delimiter, &field_count);
        free(line);
        if (!fields || field_count < 0) {
            fprintf(stderr, "csv_read: %s:%d: out of memory while parsing\n", path, line_no);
            rc = -1;
            goto done;
        }

        if (has_header && !have_header) {
            headers = fields;
            header_cols = field_count;
            have_header = 1;
            continue;
        }

        if (cols < 0) {
            cols = field_count;
            row_cap = 64;
            data = malloc(sizeof(double) * row_cap * (size_t)cols);
            if (!data) {
                fprintf(stderr, "csv_read: %s: out of memory\n", path);
                free_fields(fields, field_count);
                rc = -1;
                goto done;
            }
        } else if (field_count != cols) {
            fprintf(stderr,
                   "csv_read: %s:%d: expected %d columns (matching earlier rows), "
                   "found %d - the file must be a rectangular matrix\n",
                   path, line_no, cols, field_count);
            free_fields(fields, field_count);
            rc = -1;
            goto done;
        }

        if ((size_t)rows == row_cap) {
            row_cap *= 2;
            double *grown = realloc(data, sizeof(double) * row_cap * (size_t)cols);
            if (!grown) {
                fprintf(stderr, "csv_read: %s: out of memory\n", path);
                free_fields(fields, field_count);
                rc = -1;
                goto done;
            }
            data = grown;
        }

        for (int j = 0; j < cols; ++j) {
            double v;
            if (parse_double(fields[j], &v) != 0) {
                fprintf(stderr,
                       "csv_read: %s:%d: field %d ('%s') is not a valid number\n",
                       path, line_no, j + 1, fields[j]);
                free_fields(fields, field_count);
                rc = -1;
                goto done;
            }
            data[(size_t)rows * cols + j] = v;
        }
        free_fields(fields, field_count);
        ++rows;
    }

    if (rows == 0) {
        fprintf(stderr, "csv_read: %s: no data rows found\n", path);
        rc = -1;
        goto done;
    }
    if (has_header && header_cols != cols) {
        fprintf(stderr,
               "csv_read: %s: header has %d columns but data rows have %d\n",
               path, header_cols, cols);
        rc = -1;
        goto done;
    }

    out->data = data;   data = NULL;
    out->rows = rows;
    out->cols = cols;
    out->headers = has_header ? headers : NULL;
    headers = NULL;

done:
    fclose(f);
    free(data);
    if (headers) free_fields(headers, header_cols);
    return rc;
}

int csv_read_matrix(const char *path, int has_header, CsvMatrix *out)
{
    return csv_read(path, has_header, ',', out);
}

/* ======================================================================== */
/* Column lookup / extraction                                               */
/* ======================================================================== */

static int header_matches(const char *name, const char *target)
{
    if (!name || !target) return 0;
    for (;;) {
        unsigned char a = (unsigned char)*name, b = (unsigned char)*target;
        if (tolower(a) != tolower(b)) return 0;
        if (a == '\0') return 1;
        ++name;
        ++target;
    }
}

int csv_find_column(const CsvMatrix *m, const char *name)
{
    if (!m || !m->headers || !name) return -1;
    for (int j = 0; j < m->cols; ++j)
        if (header_matches(m->headers[j], name)) return j;
    return -1;
}

int csv_extract_column(CsvMatrix *m, int col_index, double **out_column)
{
    if (!m || !out_column || col_index < 0 || col_index >= m->cols) return -1;

    double *col = malloc(sizeof(double) * (size_t)(m->rows > 0 ? m->rows : 1));
    if (!col) return -1;
    for (int i = 0; i < m->rows; ++i)
        col[i] = m->data[(size_t)i * m->cols + col_index];

    int new_cols = m->cols - 1;

    if (new_cols == 0) {
        free(m->data);
        m->data = NULL;
        if (m->headers) {
            free(m->headers[col_index]);
            free(m->headers);
            m->headers = NULL;
        }
        m->cols = 0;
        *out_column = col;
        return 0;
    }

    double *compact = malloc(sizeof(double) * (size_t)m->rows * (size_t)new_cols);
    if (!compact) { free(col); return -1; }
    for (int i = 0; i < m->rows; ++i) {
        int dst = 0;
        for (int j = 0; j < m->cols; ++j) {
            if (j == col_index) continue;
            compact[(size_t)i * new_cols + dst++] = m->data[(size_t)i * m->cols + j];
        }
    }
    free(m->data);
    m->data = compact;

    if (m->headers) {
        free(m->headers[col_index]);
        char **new_headers = malloc(sizeof(char *) * (size_t)new_cols);
        if (new_headers) {
            int dst = 0;
            for (int j = 0; j < m->cols; ++j) {
                if (j == col_index) continue;
                new_headers[dst++] = m->headers[j];
            }
            free(m->headers);
            m->headers = new_headers;
        } else {
            /* Out of memory for the (secondary) header array: drop headers
             * entirely rather than leak the still-owned strings. The
             * extracted data itself is unaffected. */
            for (int j = 0; j < m->cols; ++j)
                if (j != col_index) free(m->headers[j]);
            free(m->headers);
            m->headers = NULL;
        }
    }

    m->cols = new_cols;
    *out_column = col;
    return 0;
}

void csv_free(CsvMatrix *m)
{
    if (!m) return;
    free(m->data);
    if (m->headers) free_fields(m->headers, m->cols);
    m->data = NULL;
    m->headers = NULL;
    m->rows = 0;
    m->cols = 0;
}

/* ======================================================================== */
/* The two MSET-facing loaders                                              */
/* ======================================================================== */

int csv_load_training(const char *path, int has_header, CsvMatrix *out)
{
    return csv_read_matrix(path, has_header, out);
}

int csv_load_test(const char *path, int has_header,
                  const char * const *time_col_candidates,
                  CsvMatrix *out, double **out_times)
{
    if (out_times) *out_times = NULL;

    int rc = csv_read_matrix(path, has_header, out);
    if (rc != 0) return rc;

    if (time_col_candidates && out_times) {
        for (int c = 0; time_col_candidates[c]; ++c) {
            int idx = csv_find_column(out, time_col_candidates[c]);
            if (idx >= 0) {
                if (csv_extract_column(out, idx, out_times) != 0) {
                    fprintf(stderr,
                           "csv_load_test: %s: failed to extract time column '%s'\n",
                           path, time_col_candidates[c]);
                }
                break;
            }
        }
    }
    return 0;
}
