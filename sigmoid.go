package math


func Sigmoid[T int | float64](z T) T {
	// Compute the sigmoid of z

	// Args:
	//     z (ndarray): A scalar, numpy array of any size.

	// Returns:
	//     g (ndarray): sigmoid(z), with the same shape as z

	// ### START CODE HERE ###
	z := 1 / (1 + math.Exp(-z))
	// ### END SOLUTION ###
	return z
}

func computeGradientLogisticReg(X []float64, y []float64, w []float64, b float64, lambda_ float64) (error, float64[], float64) {
	// Computes the gradient for linear regression

	// Args:
	//   X (ndarray (m,n): Data, m examples with n features
	//   y (ndarray (m,)): target values
	//   w (ndarray (n,)): model parameters
	//   b (scalar)      : model parameter
	//   lambda_ (scalar): Controls amount of regularization
	// Returns
	//   dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t. the parameters w.
	//   dj_db (scalar)            : The gradient of the cost w.r.t. the parameter b.

	m := len(X)
	n := len(X[0])
	dj_dw := make([]float64, n) //(n,)
	dj_db := 0.0                //scalar

	for i := 0; i < m; i++ {
		dot_p := 0.0
		for j := 0; j < n; j++ {
			dot_p += X[i][j]*w + b
		}
		f_wb_i := Sigmoid(dot_p)
		for j := 0; j < n; j++ {
			dj_dw[j] += err_i * X[i][j]
			dj_dw[j] /= m
		}
		dj_db += err_i
	}
	dj_db /= m

	for j := 0; j < n; j++ {
		dj_dw[j] += (lambda_ / m) * w[j]
	}

	return nil, dj_dw, dj_db
}

func calcSigmoid() {
	y := []float64{5.1, 4.95, 4.86, 4.73, 4.71, 4.70, 4.62, 4.39, 4.20, 3.97, 4.13, 4.21, 4.39, 4.39, 4.19, 3.83, 3.79,
    3.99, 3.89, 3.05, 3.89, 3.77, 3.99, 3.61, 3.87, 4.11, 4.05, 3.98, 3.88, 3.95, 4.21}

	X := Arange(0.0, float64(len(y)), 1.0)
	w_init := []float64{0}
	b_init := 0.0
	iterations := 10000
	tmp_alpha := 0.001
	l := 1.
	
	// gradientDescent - grdient_descent.go
	w_final, b_final, J_hist, p_hist := gradientDescent((math.Reshape(X, -1, 1) ,y, w_init, b_init, tmp_alpha, 
                                                    iterations, l, compute_cost_logistic_reg, compute_gradient_logistic_reg)
	log.Printf("(w,b) found by gradient descent: (%f,%f)", w_final, b_final)

	y1 := Sigmoid(math.Dot(X, w_final[0]) + b_final)
}
