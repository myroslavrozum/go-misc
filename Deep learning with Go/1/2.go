package main

import (
	"os"

	"log"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func two() {
	g := G.NewGraph()

	matB := []float64{0.9, 0.7, 0.4, 0.2}
	matT := tensor.New(tensor.WithBacking(matB),
		tensor.WithShape(2, 2))

	mat := G.NewMatrix(g,
		tensor.Float64,
		G.WithName("W"),
		G.WithShape(2, 2),
		G.WithValue(matT),
	)

	vecB := []float64{5, 7}
	vecT := tensor.New(tensor.WithBacking(vecB),
		tensor.WithShape(2))
	vec := G.NewVector(g,
		tensor.Float64,
		G.WithName("x"),
		G.WithShape(2),
		G.WithValue(vecT),
	)

	var z *G.Node
	var err error
	if z, err = G.Mul(mat, vec); err != nil {
		log.Fatal(err)
	}

	machine := G.NewTapeMachine(g)
	if machine.RunAll() != nil {
		log.Fatal(err)
	}
	log.Println(z.Value().Data())

	//https://dreampuf.github.io/GraphvizOnline/
	os.WriteFile("simple_graph.dot", []byte(g.ToDot()), 0644)
}
