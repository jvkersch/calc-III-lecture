# Calculus III lecture

## Deployment

First, compile the embedded apps to WebAssembly:
```console
./update-apps.sh
```
Then, publish the entire stack to GitHub pages:
```console
quarto publish gh-pages
```

## Working on the apps

Each Shiny app can be run in standalone mode from its directory:
```console
shiny run <path/to/app.py>
```