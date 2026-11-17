/* Viterbi con duraciones mínimas sobre {H, A, D}; réplica de models/decoder.py (Jan) en C para el
   laboratorio, con dos extensiones opcionales:
     - xpen[t]: coste extra (>= 0) que se resta al entrar en H desde A o en A desde H en el frame t.
     - m por clase libre (ya lo era).
   gcc -O2 -shared -fPIC -o cvit.so cvit.c
   e: T x 3 emisiones; m[3]; lq[3] log quedarse; ls[3][3] log salir (de, a); li[3] log inicio.
   out: T códigos 0/1/2. */
#include <stdlib.h>
#include <math.h>

int viterbi(const double *e, long T, const long *m, const double *lq, const double *ls,
            const double *li, const double *xpen, signed char *out)
{
    double *C = malloc(sizeof(double) * (T + 1) * 3);
    double *ent = malloc(sizeof(double) * T * 3);
    double *L = malloc(sizeof(double) * T * 3);
    signed char *arg = malloc(T * 3);
    signed char *nou = malloc(T * 3);
    long t, s;
    int c, d;
    if (!C || !ent || !L || !arg || !nou) return 1;
    for (c = 0; c < 3; c++) C[c] = 0.0;
    for (t = 0; t < T; t++)
        for (c = 0; c < 3; c++) C[(t + 1) * 3 + c] = C[t * 3 + c] + e[t * 3 + c];
    for (t = 0; t < T; t++) {
        for (c = 0; c < 3; c++) {
            if (t == 0) { ent[c] = li[c]; arg[c] = 0; }
            else {
                double best = -INFINITY; int ba = 0;
                for (d = 0; d < 3; d++) {
                    double v = L[(t - 1) * 3 + d] + ls[d * 3 + c];
                    if (xpen && d != c && d < 2 && c < 2) v -= xpen[t];
                    if (v > best) { best = v; ba = d; }
                }
                ent[t * 3 + c] = best; arg[t * 3 + c] = (signed char)ba;
            }
        }
        for (c = 0; c < 3; c++) {
            double nc = -INFINITY, q = -INFINITY, v;
            s = t - m[c];
            if (s >= 0) nc = ent[s * 3 + c] + C[t * 3 + c] - C[s * 3 + c];
            if (t > 0) q = L[(t - 1) * 3 + c] + lq[c];
            nou[t * 3 + c] = nc > q;
            v = nc > q ? nc : q;
            L[t * 3 + c] = v + e[t * 3 + c];
        }
    }
    t = T - 1;
    c = 0;
    for (d = 1; d < 3; d++) if (L[t * 3 + d] > L[t * 3 + c]) c = d;
    while (t >= 0) {
        long k = t;
        while (k >= 0 && !nou[k * 3 + c]) k--;
        if (k < 0 || !isfinite(L[t * 3 + c])) { for (s = 0; s <= t; s++) out[s] = (signed char)c; break; }
        s = k - m[c];
        { long i; for (i = s; i <= t; i++) out[i] = (signed char)c; }
        if (s <= 0) break;
        d = arg[s * 3 + c];
        c = d; t = s - 1;
    }
    free(C); free(ent); free(L); free(arg); free(nou);
    return 0;
}
