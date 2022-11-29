# snake programming language


tiny interpreted programming language with dynamic typing and just enough instuction set to implement a Snake game

### run the interpreter
``` $ cargo run --release ```

### run the game
``` $ cargo run --release -- snake_src/snake_game.snk ```

### snake code example
```rust
// calc fibonacci number

fn fib(n) {
    if n < 1 { 0 } 
    else {
        if n <= 2 { 1 }
        else { fib(n - 1) + fib(n - 2) }
    }
}

n = 20
print("fib(", n, ") = ", fib(n))
```

> heavily powered by wonderfull [rust-peg parser](https://github.com/kevinmehall/rust-peg/tree/master/tests/run-pass)


> any similarities to __python__ are purely coincidental
