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