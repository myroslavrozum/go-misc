package main

import (
	"errors"
	"log"
	"math"
)

func computeCost(x []float64, y []float64, w float64, b float64) (error, float64) {
	if len(x) != len(y) {
		return errors.New("Array sizes do not match"), math.MaxFloat64
	}

	m := len(x)
	var cost float64

	for i := 0; i < m; i++ {
		f_wb := w*x[i] + b
		cost += math.Pow((f_wb - y[i]), 2)
	}
	total_cost := 1.0 / (2.0 * float64(m)) * cost
	return nil, total_cost
}

func computeGradient(x []float64, y []float64, w float64, b float64) (error, float64, float64) {
	// Computes the gradient for linear regression
	// Args:
	//   x (ndarray (m,)): Data, m examples
	//   y (ndarray (m,)): target values
	//   w,b (scalar)    : model parameters
	// Returns
	//   dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
	//   dj_db (scalar): The gradient of the cost w.r.t. the parameter b

	// Number of training examples
	if len(x) != len(y) {
		return errors.New("Array sizes do not match"), 0, 0
	}

	m := len(x)
	var f_wb float64

	// Dot(x,w) + b
	for i := 0; i < m; i++ {
		f_wb = x[i] * w
	}
	f_wb += b

	var dj_db, dj_dw float64
	for i := 0; i < m; i++ {
		dj_db += f_wb - y[i]
		dj_dw += f_wb - x[i]
	}
	dj_db /= float64(m)
	dj_dw /= float64(m)

	return nil, dj_dw, dj_db
}

func gradientDescent(x []float64, y []float64, w_in float64, b_in float64, alpha float64, num_iters int,
	cost_function func([]float64, []float64, float64, float64) (error, float64),
	gradient_function func(x []float64, y []float64, w float64, b float64) (error, float64, float64)) (float64, float64, []float64, [][]float64) {
	// Performs gradient descent to fit w,b. Updates w,b by taking
	// num_iters gradient steps with learning rate alpha

	// Args:
	//   x (ndarray (m,))  : Data, m examples
	//   y (ndarray (m,))  : target values
	//   w_in,b_in (scalar): initial values of model parameters
	//   alpha (float):     Learning rate
	//   num_iters (int):   number of iterations to run gradient descent
	//   cost_function:     function to call to produce cost
	//   gradient_function: function to call to produce gradient

	// Returns:
	//   w (scalar): Updated value of parameter after running gradient descent
	//   b (scalar): Updated value of parameter after running gradient descent
	//   J_history (List): History of cost values
	//   p_history (list): History of parameters [w,b]

	// An array to store cost J and w's at each iteration primarily for graphing later
	J_history := make([]float64, 0)
	p_history := make([][]float64, 0)
	b := b_in
	w := w_in

	for i := 0; i < num_iters; i++ {
		// Calculate the gradient and update the parameters using gradient_function
		_, dj_dw, dj_db := gradient_function(x, y, w, b)

		// Update Parameters using equation (3) above
		b -= alpha * dj_db
		w -= alpha * dj_dw

		// Save cost J at each iteration
		if i < 100000 {
			// prevent resource exhaustion
			_, cost := cost_function(x, y, w, b)
			J_history = append(J_history, cost)
			p_history = append(p_history, []float64{w, b})
		}
		// Print cost every at intervals 10 times or as many iterations if < 10
		if i%int(math.Ceil(float64(num_iters)/10.0)) == 0 {
			log.Printf(`Iteration %4d: Cost %f dj_dw: %f, dj_db: %f  w: %f b:%f`,
				i, J_history[len(J_history)-1], dj_dw, dj_db, w, b)
		}
	}
	//return w and J,w history for graphing
	return w, b, J_history, p_history
}

// https://stackoverflow.com/questions/19906544/how-do-i-do-something-like-numpys-arange-in-go
func Arange(start, stop, step float64) []float64 {
	if step == 0 {
		return nil
	}

	size := int(math.Ceil((stop - start) / step))
	if size <= 0 {
		return []float64{}
	}

	res := make([]float64, size)
	for i := 0; i < size; i++ {
		res[i] = start + float64(i)*step
	}
	return res
}

func gradient() {
	y := []float64{5.1, 4.95, 4.86, 4.73, 4.71, 4.70, 4.62, 4.39, 4.20, 3.97, 4.13, 4.21, 4.39, 4.39, 4.19, 3.83, 3.79,
		3.99, 3.89, 3.05, 3.89, 3.77, 3.99, 3.61, 3.87, 4.11, 4.05, 3.98, 3.88, 3.95, 4.21}
	X := Arange(0.0, float64(len(y)), 0.1)

	w_init := 0.0
	b_init := 0.0
	iterations := 10000
	tmp_alpha := 0.001

	w_final, b_final, _, _ := gradientDescent(X, y, w_init, b_init, tmp_alpha, iterations, computeCost, computeGradient)
	log.Printf("(w,b) found by gradient descent: (%8.4f,%8.4f)", w_final, b_final)

}
