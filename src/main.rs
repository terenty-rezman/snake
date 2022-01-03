use ::lazy_static::lazy_static;
use ::std::collections::HashMap;
use std::f32::consts::E;
use std::os::windows::process;
use peg::error::ExpectedSet;
use std::fmt::Debug;
use std::ops::Neg;
use std::os::windows::prelude::OsStringExt;
use std::rc::Rc;

use std::{
    io::{stdout, Write},
    result,
};

#[derive(Debug)]
pub enum Expression {
    Void,
    FnArgList(Vec<Expression>),
    CodeBlock(Vec<Expression>),
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

fn bln_print(_mem: &mut Mem, args: &Vec<Rc<Object>>) -> () {
    println!("{:?}", args);
}

fn bln_mem(mem: &mut Mem, _args: &Vec<Rc<Object>>) -> () {
    println!("{:?}", mem);
}

fn bln_exit(_mem: &mut Mem, _args: &Vec<Rc<Object>>) -> () {
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

peg::parser!( grammar snake_parser() for str {
    rule _ = [' ' | '\t']*
    rule br() = ['\r' | '\n'] ['\n']?

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

        _ "{" _ br()* e:(assignment() / expression()) ** br() br()* _ "}" _ { Expression::CodeBlock(e) }

        // string
        _ "'" t:$([^'\'']+) "'" _ { Expression::String(t.to_owned()) }

        // string
        _ "\"" t:$([^'"']+) "\"" _ { Expression::String(t.to_owned()) }
    }

    rule assignment() -> Expression = _ i:ident() _ "=" _ n:expression() { Expression::Assignment(i.to_string(), n.into()) }

    pub rule program() -> Vec<Expression>
        = s:(assignment() / expression()) ** br() { s }
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

        Expression::Number(i) => Ok(Some(Rc::new(Object::Number(i)))),

        Expression::String(s) => Ok(Some(Rc::new(Object::Str(s)))),

        Expression::Ident(id) => match mem.lookup(&id) {
            Some(value) => Ok(Some(value)),
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
                    mem.insert_or_modify(&var_name, value);
                    Ok(None)
                }
            }
        }

        Expression::CodeBlock(expr_list) => {
            let mut val = None;
            mem.push_scope();
            for expr in expr_list {
                val = interpret(expr, mem)?;
            }
            mem.pop_scope();
            Ok(val)
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

fn print_parse_error(err: &peg::error::ParseError<peg::str::LineCol>, src: &str) {
    let failed_char = src.chars().nth(err.location.offset).unwrap();
    println!(
        "parse error: [line: {}, column: {}, offset: {}] expected: {} got: '{}'",
        err.location.line, err.location.column, err.location.offset, err.expected, failed_char
    );
}

use std::{collections, fs};

type Scope = HashMap<String, Rc<Object>>;

#[derive(Debug)]
struct Mem {
    mem: Vec<Scope>,
}

impl Mem {
    fn new() -> Mem {
        Mem {
            mem: vec![Scope::new()],
        }
    }

    fn push_scope(&mut self) {
        self.mem.push(Scope::new());
    }

    fn pop_scope(&mut self) {
        self.mem.pop();
    }

    fn insert_or_modify(&mut self, var_name: &str, value: Rc<Object>) {
        // if any outer scopes contain variable with given name => change that binding
        // otherwise insert value into current (deepest) scope
        match self._find_containing_scope(var_name) {
            Some((scope, _depth)) => {
                scope.insert(var_name.to_owned(), value);
            }
            None => {
                let current_scope = self.mem.last_mut().unwrap();
                current_scope.insert(var_name.to_owned(), value);
            }
        }
    }

    fn _find_containing_scope(&mut self, var_name: &str) -> Option<(&mut Scope, usize)> {
        let mut scope_depth = self.mem.len();
        for scope in self.mem.iter_mut().rev() {
            scope_depth -= 1;
            if let Some(_) = scope.get_mut(var_name) {
                return Some((scope, scope_depth));
            }
        }
        None
    }

    fn lookup(&self, var_name: &str) -> Option<Rc<Object>> {
        // iterate from deepest scope up to the global one
        // and return first variable match or None
        for scope in self.mem.iter().rev() {
            if let Some(value) = scope.get(var_name) {
                return Some(Rc::clone(value));
            }
        }
        None
    }
}

fn repl() {
    let mut mem = Mem::new();
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
        match snake_parser::program(&line) {
            Err(err) => print_parse_error(&err, &line),

            // interpret ast
            Ok(mut expr_list) => {
                // interpret 1st line
                let expr = expr_list.remove(0);

                match interpret(expr, &mut mem) {
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

use regex::Regex;

fn remove_comments_and_empty_lines(src: &str) -> String {
    // remove comments
    let re = Regex::new(r"(?m)//.*").unwrap();
    let src = re.replace_all(src, "").to_string();

    // remove empty lines
    let re = Regex::new(r"(?m)^\s*(?:\r?\n|\r)+").unwrap();
    let src = re.replace_all(&src, "").to_string();

    src.trim().to_owned()
}

#[test]
fn test_parser() {
    let src = "
        a = 1  
        {
            c = 1
        }
        b = 1
        c = a + b
        d = 1 + 1 + 1 + c
        d = {
            1
            2
        }
        print(d)
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

    let src = remove_comments_and_empty_lines(src);
    dbg!(&src);

    // try parse program into ast
    match snake_parser::program(&src) {
        Err(err) => {
            print_parse_error(&err, &src);
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

fn execute_file(src_name: String) -> Result<(), AnyError> {
    let src = fs::read_to_string(src_name)?;
    let src = remove_comments_and_empty_lines(&src);

    // try parse program into ast
    match snake_parser::program(&src) {
        Err(err) => {
            print_parse_error(&err, &src);
        }

        // interpret each expr in ast
        Ok(ast) => {
            let mut memory = Mem::new();
            for statement in ast {
                interpret(statement, &mut memory).unwrap();
            }
        }
    }

    Ok(())
}

use std::env;

fn main() {
    let mut args = env::args();
    if args.len() > 1 {
        let src = args.nth(1).unwrap();
        if let Err(e) = execute_file(src) {
            println!("\nError: {}", e);
            std::process::exit(1);
        }
    }
    else{
        repl();
    }
}
