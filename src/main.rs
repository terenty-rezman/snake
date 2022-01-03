use ::lazy_static::lazy_static;
use peg::error::ExpectedSet;
use ::std::collections::HashMap;
use std::rc::Rc;
use std::ops::Neg;

use std::{
    io::{stdout, Write},
    result,
};

#[derive(Debug)]
pub enum Expression {
    Void,
    FnArgList(Vec<Expression>),
    Ident(String),
    Number(i64),
    String(String),
    Assignment(String, Box<Expression>),
    Add(Box<Expression>, Box<Expression>),
    Sub(Box<Expression>, Box<Expression>),
    Mul(Box<Expression>, Box<Expression>),
    Div(Box<Expression>, Box<Expression>),
    Pow(Box<Expression>, Box<Expression>),
    Neg(Box<Expression>),
    FnCall(String, Box<Expression>),
}

type DynamicFcn = fn(&mut Mem, &Vec<Rc<Object>>) -> ();

fn bln_print(mem: &mut Mem, args: &Vec<Rc<Object>>) -> () {
    println!("{:?}", args);
}

fn bln_mem(mem: &mut Mem, args: &Vec<Rc<Object>>) -> () {
    println!("{:?}", mem);
}

fn bln_exit(mem: &mut Mem, args: &Vec<Rc<Object>>) -> () {
    println!("sayōnara");
    std::process::exit(0);
}

lazy_static! {
    static ref BUILTINS: HashMap<String, DynamicFcn> = {
        let mut m = HashMap::new();
        m.insert("print".to_owned(), bln_print as DynamicFcn);
        m.insert("mem".to_owned(), bln_mem as DynamicFcn);
        m.insert("exit".to_owned(), bln_exit as DynamicFcn);
        m
    };
}

peg::parser!( grammar hiki_parser() for str {
    rule _ = " "*
    rule br() = "\n"*

    pub rule ident() -> &'input str
        = s: $(['a'..='z' | 'A'..='Z']['_' | 'a'..='z' | 'A'..='Z' | '0'..='9']*) {
            s
        }

    rule number() -> i64
        = n: $(['0'..='9']+) { n.parse().unwrap() }

    pub rule expression() -> Expression = precedence! {
        // x + y
        x:(@) "+" y:@ { Expression::Add(x.into(), y.into()) }
        // x - y
        x:(@) "-" y:@ { Expression::Sub(x.into(), y.into()) }
        --
        // x * y
        x:(@) "*" y:@ { Expression::Mul(x.into(), y.into()) }
        // y / y
        x:(@) "/" y:@ { Expression::Div(x.into(), y.into()) }
          --
        x:@ "^" y:(@) { Expression::Pow(x.into(), y.into()) }
        "-" x:@ { Expression::Neg(x.into()) }
        --
        // fn call
        _ f:ident() "(" e:expression() ** "," ")" _ {
            Expression::FnCall(f.to_owned(), Expression::FnArgList(e).into())
        }

        // identificator
        _ i:ident() _ { Expression::Ident(i.to_string()) }

        // number
        _ n:number() _ { Expression::Number(n) }

        _ "(" e:expression() ")" _ { e }

        // string
        _ "'" t:$([^'\'']+) "'" _ { Expression::String(t.to_owned()) }

        // string
        _ "\"" t:$([^'"']+) "\"" _ { Expression::String(t.to_owned()) }
    }

    rule assignment() -> Expression = _ i:ident() _ "=" _ n:expression() { Expression::Assignment(i.to_string(), n.into()) }

    pub rule program() -> Vec<Expression>
        = br() _ s:(assignment() / expression()) ** "\n" br() _ { s }
});

type AnyError = Box<dyn std::error::Error>;
type EvalResult = Result<Option<Rc<Object>>, AnyError>;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum EvalError {
    #[error("{0}")]
    ValueError(String),
}

use EvalError::*;

#[derive(Debug)]
enum Object {
    Str(String),
    Number(i64),
    Array(Vec<Rc<Object>>),
}

fn eval_args_and_do_ariphmetic(
    x: Expression,
    y: Expression,
    op: fn(i64, i64) -> i64,
    mem: &mut Mem,
) -> EvalResult {
    let x = interpret(x, mem)?.ok_or(ValueError("value expected".to_owned()))?;
    let y = interpret(y, mem)?.ok_or(ValueError("value expected".to_owned()))?;

    let x = match *x {
        Object::Number(n) => n,
        _ => Err(ValueError("integer expected".to_owned()))?,
    };

    let y = match *y {
        Object::Number(n) => n,
        _ => Err(ValueError("integer expected".to_owned()))?,
    };

    let result = op(x, y);

    Ok(Some(Object::Number(result).into()))
}

fn interpret(exp: Expression, mem: &mut Mem) -> EvalResult {
    match exp {
        Expression::Void => Ok(None),

        Expression::Number(i) => Ok(Some(Object::Number(i).into())),

        Expression::String(s) => Ok(Some(Object::Str(s).into())),

        Expression::Ident(id) => match mem.get(&id) {
            Some(value) => Ok(Some(value.clone())),
            None => Err(ValueError(format!("'{}' variable not found", id)).into()),
        },

        Expression::FnArgList(v) => {
            let mut results = Vec::new();
            for e in v {
                let r = interpret(e, mem)?.ok_or(ValueError("value expected".to_owned()))?;
                results.push(r);
            }

            Ok(Some(Object::Array(results).into()))
        }

        Expression::Assignment(var_name, expr) => {
            let value = interpret(*expr, mem)?;
            match value {
                None => Err(ValueError("value expected".to_owned()).into()),
                Some(value) => {
                    mem.insert(var_name, value);
                    Ok(None)
                }
            }
        }

        Expression::Add(expr_1, expr_2) => {
            eval_args_and_do_ariphmetic(*expr_1, *expr_2, std::ops::Add::add, mem)
        }

        Expression::Sub(expr_1, expr_2) => {
            eval_args_and_do_ariphmetic(*expr_1, *expr_2, std::ops::Sub::sub, mem)
        }

        Expression::Mul(expr_1, expr_2) => {
            eval_args_and_do_ariphmetic(*expr_1, *expr_2, std::ops::Mul::mul, mem)
        }

        Expression::Div(expr_1, expr_2) => {
            eval_args_and_do_ariphmetic(*expr_1, *expr_2, std::ops::Div::div, mem)
        }

        Expression::Pow(expr_1, expr_2) => {
            eval_args_and_do_ariphmetic(*expr_1, *expr_2, |x, y| x.pow(y as u32), mem)
        }

        Expression::Neg(expr_1) => {
            eval_args_and_do_ariphmetic(*expr_1, Expression::Number(0), |x, _| x.neg(), mem)
        }

        Expression::FnCall(fn_name, expr) => {
            let value = interpret(*expr, mem)?.ok_or(ValueError("arg list expected".to_owned()))?;

            let builtin = BUILTINS
                .get(&fn_name)
                .ok_or(ValueError(format!("no such builtin found: '{}'", fn_name)))?;

            let vec = match &*value {
                Object::Array(v) => v,
                _ => Err(ValueError("integer expected".to_owned()))?,
            };

            builtin(mem, &vec);
            Ok(None)
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

type Mem = std::collections::HashMap<String, Rc<Object>>;

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

                match interpret(expr, mem) {
                    Err(err) => println!("Error: {}", &err),
                    Ok(result) => {
                        if let Some(result) = result {
                            println!("{:?}", result);
                        }
                    }
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
        d = a - b
        d = a * b
        d = a / b
        d = a ^ b
        d = -1
        print(a + a)
        print('hello world')
        s = 'hello'
        print(s)
        (2 + 3) + 3
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
                interpret(statement, &mut memory).unwrap();
            }
            dbg!(memory);
        }
    }
}

fn main() {
    let mut memory = Mem::new();
    repl(&mut memory);
}
