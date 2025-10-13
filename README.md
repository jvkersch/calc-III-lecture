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

It is also possible to preview the finalized output before pushing to GitHub pages. In this case, all resources will be served locally (perfect for working on the plane):
```console
quarto preview
```

## Working on the apps

For easy local development, each Shiny app can be run in standalone mode from its directory:
```console
shiny run <shiny-apps/path/to/app.py>
```

As the apps are pure Python files, running the linter is easy:
```console
ruff check --fix
```