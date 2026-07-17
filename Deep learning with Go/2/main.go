package main

import (
	"log"
	"os"
	"os/signal"
	"syscall"

	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type nn struct {
	g      *ExprGraph
	w0, w1 *Node

	pred *Node
}

func (m *nn) learnables() Nodes {
	return Nodes{m.w0}
}

func (m *nn) fwd(x *Node) (err error) {
	var l0, l1 *Node

	// Set first layer to be copy of input
	l0 = x

	// Dot product of l0 ans w0, use as input for Sigmoid
	l0dot := Must(Mul(l0, m.w0))

	// Build hidden layer out of result
	l1 = Must(Sigmoid(l0dot))
	log.Println("l1: \n", l1.Value())

	if l1 != nil {
		m.pred = l1
	}

	return
}

func newNN(g *ExprGraph) *nn {
	// Create node for w/weight (needs fixed values replaced) with random values w/mean 0)
	wB := []float64{-0.168855599, 0.44064899, -0.99977125}
	wT := tensor.New(tensor.WithBacking(wB),
		tensor.WithShape(3, 1))
	w0 := NewMatrix(g,
		tensor.Float64,
		WithName("W"),
		WithShape(3, 1),
		WithValue(wT))

	return &nn{
		g:  g,
		w0: w0,
	}

}
func main() {
	//rand.Seed(31337)

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// doneChan := make(chan bool, 1)

	// Create graph and network
	g := NewGraph()
	m := newNN(g)

	xB := []float64{0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1}
	xT := tensor.New(tensor.WithBacking(xB),
		tensor.WithShape(4, 3))

	x := NewMatrix(g,
		tensor.Float64,
		WithName("X"),
		WithShape(4, 3),
		WithValue(xT))

	// Define validation dataset
	yB := []float64{0, 0, 1, 1}
	yT := tensor.New(tensor.WithBacking(yB),
		tensor.WithShape(4, 1))
	y := NewMatrix(g,
		tensor.Float64,
		WithName("y"),
		WithShape(4, 1),
		WithValue(yT))

	var err error
	if err = m.fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}

	losses := Must(Sub(y, m.pred))
	square := Must(Square(losses))
	cost := Must(Mean(square))

	var costVal Value
	Read(cost, &costVal)

	var predVal Value
	Read(m.pred, &predVal)

	//https://dreampuf.github.io/GraphvizOnline/
	os.WriteFile("pregrad.dot", []byte(g.ToDot()), 0644)

	if _, err = Grad(cost, m.learnables()...); err != nil {
		log.Fatal(err)
	}

	// Instantiate VM and Solver
	vm := NewTapeMachine(g, BindDualValues(m.learnables()...))
	solver := NewVanillaSolver(WithLearnRate(2.0), WithClip(5))
	// solver := NewRMSPropSolver()

	for i := range 10000 {
		if err = vm.RunAll(); err != nil {
			log.Fatalf("Failed at iter %d: %v", i, err)
		}
		solver.Step(NodesToValueGrads(m.learnables()))
		log.Println("\nState at iter", i)
		log.Println("Cost: \n", cost.Value())
		log.Println("Weights: \n", m.w0.Value())
		// vm.Set(m.w0, wUpd)
		vm.Reset()
	}
	vm.RunAll()
	log.Println("Output adter Training: \n", predVal)
}
