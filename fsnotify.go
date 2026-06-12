package main

import (
	"log"

	"github.com/fsnotify/fsnotify"
)

// Credits: https://github.com/fsnotify/fsnotify
func dirwatcher() {
	// Create new watcher.
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		log.Fatal(err)
	}
	defer watcher.Close()

	// Start listening for events.
	go func() {
		for {
			select {
			case event, ok := <-watcher.Events:
				if !ok {
					return
				}
				log.Printf("Event: %s (Op: %s)", event.Name, event.Op)

				if event.Has(fsnotify.Write) {
					log.Println("Modified file:", event.Name)
				}
			case err, ok := <-watcher.Errors:
				if !ok {
					return
				}
				log.Println("Error:", err)
			}
		}
	}()

	// Add the directory you want to watch.
	err = watcher.Add("./")
	if err != nil {
		log.Fatal(err)
	}

	// Block the main goroutine so it stays open.
	<-make(chan struct{})
}
