#include "kalman.h"

void kalman_prediction();
void kalman_update();
//void filter(void (*prediction_fcn)(MAT *), void update_fcn, MAT *x, MAT *y);

int main(void **argv){

    MAT *A, *Q, *R, *P, *H;
    VEC *x, *y; 

    A = m_get(N_STATES, N_STATES);
    m_ident(A);
    sm_mlt(0.5, A, A);

    Q = m_get(N_STATES, N_STATES);
    m_ident(Q);
    R = m_get(N_STATES, N_STATES);
    m_ident(R);
    P = m_get(N_STATES, N_STATES);
    m_ident(P);
    H = m_get(N_MEASUREMENTS, N_STATES);
    m_ident(H);

    x = v_get(N_STATES);
    v_ones(x);
    y = v_get(N_MEASUREMENTS);
    
    //v_output(mv_mlt(A,x, VNULL));
    m_output(P);
    kalman_prediction(x, P, A, Q);
    m_output(P);
    kalman_update(x, P, y, H, R);
    m_output(P);

    FILE *fp;
    fp = fopen("test.mat", "w");
    char name[] = {'P'};
    m_save(fp, P, name);
}

/**
 * Computes the linear Kalman filter prediction step.
 * 
 * 
 * @param VEC *x the prior state density mean.
 * @param MAT *P the prior state density covariance.
 * @param MAT *A the linear process model.
 * @param MAT *Q the process noise covariance
 * 
 * @return Changes x, P into the predicted density  
 */
void kalman_prediction(VEC *x, MAT *P, const MAT *A, const MAT *Q){
    // Xkp1 = Ax
    // Pkp1 = APA + Q
    v_copy(mv_mlt(A, x, VNULL), x);
    m_add(m_mlt(m_mlt(A, P, MNULL), A, P), Q, P);
}

/**
 * Computes the linear Kalman filter update step.
 * 
 * 
 * @param VEC *x the prior state density mean.
 * @param MAT *P the prior state density covariance.
 * @param MAT *H the linear measurement model.
 * @param MAT *R the measurement noise covariance
 * 
 * @return Changes x, P into the predicted density  
 */
void kalman_update(VEC *x, MAT *P, VEC *y, const MAT *H, const MAT *R){
    // S = HPH' + R
    // K = PH'inv(s)
    // P = P - KHP
    // x = x + K(y - Hx)
    MAT *S, *K, *m_tmp;

    S = m_add(mmtr_mlt(m_mlt(H, P, MNULL), H, MNULL), R, MNULL);

    K = m_inverse(S, MNULL);
    m_sub(P, m_mlt(m_mlt(K, H, MNULL), P, MNULL), P);
    v_add(x, mv_mlt(K, v_sub(y, mv_mlt(H, x, VNULL), VNULL), VNULL), x);

}

/**
 * Runs filtering on state x given measurements y
 * 
 * @param void prediction_fcn the prediction step function.
 * @param void update_fcn the update step function.
 * @param MAT *x the state trajectory
 * @param MAT *y measurements
 * 
 */
void filter(void (*prediction_fcn)(MAT *x), void (*update_fcn)(MAT *x), MAT *x, MAT *y){
    puts("Not implemented!");
    //int N = y->m;

}