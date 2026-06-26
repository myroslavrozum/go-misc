package main

import "flag"

func main() {
	asyncFlag := flag.Bool("async", false, "Run asynchronous whoth channels example")
	dirWatcherFlag := flag.Bool("dirwatcher", false, "Run fsnotify example")
	gradientFlag := flag.Bool("gradient", false, "Run gradient descent example")

	flag.Parse()

	if asyncFlag != nil && *asyncFlag {
		async()
	}

	if dirWatcherFlag != nil && *dirWatcherFlag {
		dirwatcher()
	}

	if gradientFlag != nil && *gradientFlag {
		gradient()
	}
}
