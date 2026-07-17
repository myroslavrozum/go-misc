package main

import (
	"log"

	"gorgonia.org/gorgonia"
)

func one() {

	var a, b, c *gorgonia.Node
	var err error

	g := gorgonia.NewGraph()

	// Define the expression
	a = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("a"))
	b = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("b"))

	if c, err = gorgonia.Add(a, b); err != nil {
		log.Fatal(err)
	}

	// Create the VM to run the program on
	machine := gorgonia.NewTapeMachine(g)

	gorgonia.Let(a, 1.0)
	gorgonia.Let(b, 2.0)
	if machine.RunAll() != nil {
		log.Fatal(err)
	}

	log.Printf("%v", c.Value())
}
