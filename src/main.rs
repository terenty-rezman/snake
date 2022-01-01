use ::lazy_static::lazy_static;
use ::std::collections::HashMap;
use std::{
    io::{stdout, Write},
    result,
};

#[derive(Debug)]
pub enum Expression {
    Void,
    Ident(String),
    Number(i64),
    String(String),
    Assignment(String, Box<Expression>),
    Add(Box<Expression>, Box<Expression>),
    FnCall(String, Box<Expression>),
}

fn bln_print(s: String) -> () {
    println!("{}", s);
}

lazy_static! {
    static ref BUILTINS: HashMap<String, fn(String) -> ()> = {
        let mut m = HashMap::new();
        m.insert("print".to_owned(), bln_print as fn(String) -> ());
        m
    };
}

peg::parser!( grammar hiki_parser() for str {
    rule _ = " "*
    rule br() = "\n"*

    rule plus() = "+"

    pub rule ident() -> &'input str
        = s: $(['a'..='z' | 'A'..='Z']['_' | 'a'..='z' | 'A'..='Z' | '0'..='9']*) {
            s
        }

    rule number() -> Expression
        = n: $(['0'..='9']+) { Expression::Number(n.parse().unwrap()) }

    pub rule expression() -> Expression = precedence! {
        // a + b
        x:(@) plus() y:@ {
            Expression::Add(x.into(), y.into())
        }

        // fn call
        _ f:ident() "(" e:expression() ")" _ { Expression::FnCall(f.to_owned(), e.into())}

        --
        // assignment
        _ i:ident() _ "=" _ n:expression() { Expression::Assignment(i.to_string(), n.into()) }

        // identificator
        _ i:ident() _ { Expression::Ident(i.to_string()) }

        // number
        _ n:number() _ { n }

        // string
        _ "'" t:$(['a'..='z' | 'A'..='Z' | ' ']+) "'" _ { Expression::String(t.to_owned()) }

        // void
        // _ "" _ { Expression::Void }
    }

    pub rule program() -> Vec<Expression>
        = br() _ s:expression() ** "\n" br() _ { s }
});

fn interpret(exp: Expression, mem: &mut Mem) -> Option<String> {
    match exp {
        Expression::Number(i) => Some(i.to_string()),

        Expression::String(s) => Some(s),

        Expression::Ident(id) => {
            let value = mem
                .get(&id)
                .expect(format!("'{}' variable not found", id).as_str());
            Some(value.clone())
        }

        Expression::Assignment(var_name, expr) => {
            let value = interpret(*expr, mem).unwrap();
            mem.insert(var_name, value);
            None
        }

        Expression::Add(expr_1, expr_2) => {
            let value_1 = interpret(*expr_1, mem).expect("value expected");
            let value_2 = interpret(*expr_2, mem).expect("value expected");

            Some((value_1.parse::<i64>().unwrap() + value_2.parse::<i64>().unwrap()).to_string())
        }

        Expression::FnCall(fn_name, expr) => {
            let value = interpret(*expr, mem).expect("value expected");

            let builtin = BUILTINS.get(&fn_name).expect("no such builtin found");
            builtin(value);
            None
        }
        _ => unimplemented!(),
    }
}

fn print_parse_error(err: &peg::error::ParseError<peg::str::LineCol>) {
    println!(
        "parse error: [line: {}, column: {}, offset: {}] expected: {}",
        err.location.line, err.location.column, err.location.offset, err.expected
    );
}

type Mem = std::collections::HashMap<String, String>;

fn repl(mem: &mut Mem) {
    println!("\n💩 v0.1\n");

    loop {
        print!("> ");
        std::io::stdout().flush().unwrap();

        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();
        line = line.trim_end().to_owned();

        if line.is_empty() {
            continue;
        }

        // try line program into ast and then interpret
        match hiki_parser::program(&line) {
            Err(err) => print_parse_error(&err),

            // interpret ast
            Ok(mut expr_list) => {
                // interpret 1st line
                let expr = expr_list.remove(0);
                if let Some(result) = interpret(expr, mem) {
                    println!("{}", result);
                }
            }
        }
    }
}

#[test]
fn test_parser() {
    let src = "
        a = 1
        b = 1
        c = a + b
        d = 1 + 1 + 1 + c
        print(a + a)
        print('hello world')
        s = 'hello'
        print(s)
    ";

    // try parse program into ast
    match hiki_parser::program(src) {
        Err(err) => {
            print_parse_error(&err);
            panic!();
        }

        // interpret each expr in ast
        Ok(ast) => {
            let mut memory = Mem::new();
            
            for statement in ast {
                interpret(statement, &mut memory);
            }
            dbg!(memory);
        }
    }
}

fn main() {
    let mut memory = Mem::new();
    repl(&mut memory);
}
