#include <stdio.h>
#include <matrix.h>

#define n 2
#define m 2
#define o 2

void kalman_prediction();

int main(void **argv){

    MAT *A, *Q, *R, *P, *P_new;
    VEC *x, *x_new; 

    A = m_get(n,n);
    m_ident(A);
    sm_mlt(0.5, A, A);

    Q = m_get(n,n);
    m_ident(Q);
    P = m_get(n,n);
    m_ident(P);
    P_new = m_get(n,n);

    x = v_get(n);
    v_ones(x);
    x_new = v_get(n);
    
    //v_output(mv_mlt(A,x, VNULL));
    kalman_prediction(x, P, x_new, P_new, A, Q);
    v_output(x);
    v_output(x_new);
    m_output(P);
    m_output(P_new);
}

void kalman_prediction(VEC *x, MAT *P, VEC *x_new, MAT *P_new, MAT *A, MAT *Q){
    // Xkp1 = Ax
    // Pkp1 = APA + Q
    MAT *P_tmp;
    P_tmp = m_get(n, n);

    mv_mlt(A, x, x_new);

    m_mlt(A, P, P_tmp);
    m_mlt(P_tmp, A, P_new);
    m_add(P_new, Q, P_new);

    m_free(P_tmp);
}

