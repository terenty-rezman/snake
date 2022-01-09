use ::lazy_static::lazy_static;
use ::std::collections::HashMap;
use crossterm::queue;
use std::fmt::Debug;
use std::io::Stdout;
use std::ops::Neg;
use std::rc::Rc;

use std::io::{stdout, Write};

// have to implement wrapper type and Debug trait manually on the wrapper because of the fcn pointer with lifetimes
// https://users.rust-lang.org/t/impl-of-debug-is-not-general-enough-error/64284
#[derive(Clone)]
pub struct OrdFcn(fn(&Object, &Object) -> bool);

impl Debug for OrdFcn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", std::any::type_name::<Self>())
    }
}

#[derive(Clone, Debug)]
pub enum Expression {
    FnDef {
        name: String,
        arg_names: Vec<String>,
        body: Box<Expression>,
    },

    FnCall {
        name: String,
        args: Vec<Expression>,
    },

    CodeBlock(Vec<Expression>),

    IfElseBlock {
        cond: Box<Expression>,
        true_block: Box<Expression>,
        else_block: Box<Expression>,
    },

    WhileBlock {
        cond: Box<Expression>,
        body: Box<Expression>,
    },

    CmpOperator {
        ord_fn: OrdFcn,
        lhs: Box<Expression>,
        rhs: Box<Expression>,
    },

    EqOperator(Box<Expression>, Box<Expression>),
    NotEqOperator(Box<Expression>, Box<Expression>),
    Ident(String),
    Number(i64),
    String(String),
    ArrayLiteral(Vec<Expression>),

    ArrayIndexing {
        arr_name: String,
        expr: Box<Expression>,
    },

    Assignment {
        var_name: String,
        expr: Box<Expression>,
    },

    Add(Box<Expression>, Box<Expression>),
    Sub(Box<Expression>, Box<Expression>),
    Mul(Box<Expression>, Box<Expression>),
    Div(Box<Expression>, Box<Expression>),
    Pow(Box<Expression>, Box<Expression>),
    Neg(Box<Expression>),
}

type BuiltinFcn = fn(&mut Stdout, &mut Mem, &Vec<Rc<Object>>) -> EvalResult;

fn snk_print(_stdout: &mut Stdout, _mem: &mut Mem, args: &Vec<Rc<Object>>) -> EvalResult {
    for (i, o) in args.iter().enumerate() {
        if i > 0 {
            print!(" ");
        }
        print!("{}", &o);
    }
    print!("\n");
    std::io::stdout().flush().unwrap();

    Ok(None)
}

fn snk_mem(_stdout: &mut Stdout, mem: &mut Mem, _args: &Vec<Rc<Object>>) -> EvalResult {
    println!("{:?}", mem);
    Ok(None)
}

fn snk_exit(_stdout: &mut Stdout, _mem: &mut Mem, _args: &Vec<Rc<Object>>) -> EvalResult {
    println!("bye");
    std::process::exit(0);
}

fn snk_len(_stdout: &mut Stdout, _mem: &mut Mem, args: &Vec<Rc<Object>>) -> EvalResult {
    if args.is_empty() {
        Err("len(x) expects one argument")?
    }

    match &*args[0] {
        Object::Array(v) => Ok(Some(Rc::new(Object::Number(v.len() as i64)))),
        _ => Err("len(x) expects array as an input")?,
    }
}

fn snk_array_push(_stdout: &mut Stdout, _mem: &mut Mem, args: &Vec<Rc<Object>>) -> EvalResult {
    if args.len() < 2 {
        Err("push() expects 2 arguments")?
    }

    match args[0].as_ref() {
        Object::Array(v) => {
            let value = &args[1];
            let mut new_vec = v.clone();
            new_vec.push(Rc::clone(value));
            Ok(Some(Rc::new(Object::Array(new_vec))))
        }
        _ => Err("push(array, value) expects array as an input")?,
    }
}

fn snk_array_push_front(
    _stdout: &mut Stdout,
    _mem: &mut Mem,
    args: &Vec<Rc<Object>>,
) -> EvalResult {
    if args.len() < 2 {
        Err("push_front() expects 2 arguments")?
    }

    match args[0].as_ref() {
        Object::Array(v) => {
            let value = &args[1];
            let mut new_vec = v.clone();
            new_vec.insert(0, Rc::clone(value));
            Ok(Some(Rc::new(Object::Array(new_vec))))
        }
        _ => Err("push(array, value) expects array as an input")?,
    }
}

fn snk_array_pop(_stdout: &mut Stdout, _mem: &mut Mem, args: &Vec<Rc<Object>>) -> EvalResult {
    if args.len() < 1 {
        Err("pop() expects 1 argument")?
    }

    match args[0].as_ref() {
        Object::Array(v) => {
            let mut new_vec = v.clone();
            new_vec.pop();
            Ok(Some(Rc::new(Object::Array(new_vec))))
        }
        _ => Err("pop(array) expects array as an input")?,
    }
}

lazy_static! {
    static ref BUILTINS: HashMap<String, BuiltinFcn> = {
        let mut m = HashMap::new();
        m.insert("print".to_owned(), snk_print as BuiltinFcn);
        m.insert("mem".to_owned(), snk_mem as BuiltinFcn);
        m.insert("exit".to_owned(), snk_exit as BuiltinFcn);
        m.insert("len".to_owned(), snk_len as BuiltinFcn);
        m.insert("push".to_owned(), snk_array_push as BuiltinFcn);
        m.insert("push_front".to_owned(), snk_array_push_front as BuiltinFcn);
        m.insert("pop".to_owned(), snk_array_pop as BuiltinFcn);
        m.insert("rand_int".to_owned(), snk_rand_int as BuiltinFcn);

        m.insert(
            "enter_game_mode".to_owned(),
            snk_enter_game_mode as BuiltinFcn,
        );
        m.insert(
            "leave_game_mode".to_owned(),
            snk_leave_game_mode as BuiltinFcn,
        );
        m.insert("poll_event".to_owned(), snk_poll_event as BuiltinFcn);
        m.insert("screen_size".to_owned(), snk_screen_size as BuiltinFcn);
        m.insert("clear_screen".to_owned(), snk_clear_screen as BuiltinFcn);
        m.insert("print_at_pos".to_owned(), snk_print_at_pos as BuiltinFcn);
        m.insert("flush".to_owned(), snk_flush as BuiltinFcn);
        m
    };
}

fn snk_rand_int(_stdout: &mut Stdout, _mem: &mut Mem, args: &Vec<Rc<Object>>) -> EvalResult {
    if args.is_empty() {
        Err("rand_int(max) expects one argument")?
    }

    let max = match &*args[0] {
        Object::Number(n) => *n,
        _ => Err("rand_int(max) expects integer as an input")?,
    };

    use rand::Rng;
    let mut rng = rand::thread_rng();
    Ok(Some(Rc::new(Object::Number(rng.gen_range(0..max)))))
}

peg::parser!( grammar snake_parser() for str {
    // whitespace rules taken from https://gist.github.com/zicklag/aad1944ef7f5dd256218892477f32c64
    // Whitespace character
    rule whitespace_char() = ['\t' | ' ' | ';']

    // Line comment
    rule line_comment() = "//" (!"\n" [_])* ("\n" / ![_])

    // Whitespace including comments
    rule _ = quiet!{ (whitespace_char() / line_comment())* }

    // Whitespace including newlines and line comments
    rule wn() = quiet!{ (whitespace_char() / "\n" / line_comment())* }

    pub rule ident() -> &'input str
        = _ s: $(['a'..='z' | 'A'..='Z']['_' | 'a'..='z' | 'A'..='Z' | '0'..='9']*) { s }

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
        _ f:ident() "(" l:anything() ** "," ")" _ {
            Expression::FnCall {
                name: f.to_owned(), args:
                l.into()
            }
        }

        // array literal
        _ "[" a:anything() ** "," "]" _ {
            Expression::ArrayLiteral(a)
        }

        // array indexing
        _ i:ident()"[" e:anything() "]" _ {
            Expression::ArrayIndexing {
                arr_name: i.to_string(),
                expr: e.into()
            }
        }

        // identificator
        _ i:ident() _ { Expression::Ident(i.to_string()) }

        // number
        _ n:number() _ { Expression::Number(n) }

        _ "(" e: (cmp_ops() / expression()) ")" _ { e }

        // string
        _ "'" t:$([^'\'']+) "'" _ { Expression::String(t.to_owned()) }

        // string
        _ "\"" t:$([^'"']+) "\"" _ { Expression::String(t.to_owned()) }
    }

    rule less() -> Expression
        = _ x:expression() "<" y:expression() {
        Expression::CmpOperator {
            ord_fn: OrdFcn(std::cmp::PartialOrd::lt),
            lhs: x.into(),
            rhs: y.into()
        }
    }

    rule less_eq() -> Expression
        = _ x:expression() "<=" y:expression() {
        Expression::CmpOperator {
            ord_fn: OrdFcn(std::cmp::PartialOrd::le),
            lhs: x.into(),
            rhs: y.into()
        }
    }

    rule greater() -> Expression
        = _ x:expression() ">" y:expression() {
        Expression::CmpOperator{
            ord_fn: OrdFcn(std::cmp::PartialOrd::gt),
            lhs: x.into(),
            rhs: y.into()
        }
    }

    rule greater_eq() -> Expression
        = _ x:expression() ">=" y:expression() {
        Expression::CmpOperator {
            ord_fn: OrdFcn(std::cmp::PartialOrd::gt), lhs:
            x.into(), rhs:
            y.into()
        }
    }

    rule eq() -> Expression
        = _ x:expression() "==" y:expression() {
        Expression::EqOperator(x.into(), y.into())
    }

    rule not_eq() -> Expression
        = _ x:expression() "!=" y:expression() {
        Expression::NotEqOperator(x.into(), y.into())
    }

    rule cmp_ops() -> Expression = less() / less_eq() / greater() / greater_eq() / eq() / not_eq()

    rule if_else() -> Expression
        = _ "if" _ c:(cmp_ops() / expression()) wn() t:code_block() wn() e:else_block()? {
        Expression::IfElseBlock {
            cond: c.into(),
            true_block: t.into(),
            else_block: e.unwrap_or(Expression::CodeBlock(Vec::new()).into()).into()
        }
    }

    rule else_block() -> Expression = _ "else" wn() e:code_block() { e }

    rule while_block() -> Expression
        = _ "while" _ c:(cmp_ops() / expression()) wn() b:code_block() {
            Expression::WhileBlock {
                cond: c.into(),
                body: b.into()
            }
        }

    rule code_block() -> Expression
        = _ "{" wn() e:(fn_def() / anything()) ** wn() wn() "}" _ {
            Expression::CodeBlock(e)
        }

    rule assignment() -> Expression
        = _ i:ident() _ "=" _ n:(code_block() / if_else() / cmp_ops() / expression()) {
            Expression::Assignment {
                var_name: i.to_string(),
                expr: n.into()
            }
        }

    rule fn_def() -> Expression
        = _ "fn" _ i:ident() "(" l:ident() ** "," ")" wn() b:code_block() {
            Expression::FnDef {
                name: i.to_string(),
                arg_names: l.into_iter().map(|s| s.to_string()).collect(),
                body: b.into()
            }
        }

    rule anything() -> Expression = assignment() / code_block() / if_else() / while_block() / cmp_ops() / expression()

    pub rule program() -> Vec<Expression>
        = wn() s:(fn_def() / anything()) ** wn() wn() { s }
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

// need to provide Ast wrapper for Expression to manually define Ord cmp traits
#[derive(Debug)]
struct Ast(Expression);

impl PartialEq for Ast {
    fn eq(&self, _other: &Ast) -> bool {
        false
    }
}

impl Eq for Ast {}

impl PartialOrd for Ast {
    fn partial_cmp(&self, _other: &Self) -> Option<std::cmp::Ordering> {
        unreachable!();
    }
}

impl Ord for Ast {
    fn cmp(&self, _other: &Self) -> std::cmp::Ordering {
        unreachable!();
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Object {
    Str(String),
    Number(i64),
    Array(Vec<Rc<Object>>),
    Function(Vec<String>, Ast), // need Ast wrapper to use autoderive cmp traits
}

impl From<&str> for Object {
    fn from(string: &str) -> Self {
        Object::Str(string.to_owned())
    }
}

impl From<i64> for Object {
    fn from(n: i64) -> Self {
        Object::Number(n)
    }
}

macro_rules! object_vec {
    ( $( $x:expr ),* ) => {
        {
            let mut obj_vec = Vec::<Rc<Object>>::new();
            $(
                obj_vec.push(Rc::new($x.into()));
            )*
            obj_vec
        }
    };
}

impl Object {
    fn eval_to_bool(&self) -> Result<bool, AnyError> {
        match self {
            Object::Str(s) => Ok(!s.is_empty()),
            Object::Number(n) => Ok(*n != 0),
            Object::Array(v) => Ok(v.len() != 0),
            _ => Err(ValueError(format!("{:?} can not eval to bool", self)).into()),
        }
    }
}

impl std::fmt::Display for Object {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Object::Str(s) => write!(f, "{}", s),
            Object::Number(n) => write!(f, "{}", n),
            Object::Array(v) => {
                write!(f, "[")?;
                v.iter().enumerate().for_each(|(i, o)| {
                    if i != 0 {
                        write!(f, ",");
                    }
                    write!(f, "{}", o);
                });
                write!(f, "]")
            }
            _ => write!(f, "{:?}", self),
        }
    }
}

fn eval_args_and_do_ariphmetic(
    x: &Expression,
    y: &Expression,
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

fn try_call_fn(fcn: &Object, arg_values: &Vec<Rc<Object>>, mem: &mut Mem) -> EvalResult {
    match fcn {
        Object::Function(arg_names, body) => {
            if arg_values.len() < arg_names.len() {
                Err(ValueError(format!(
                    "function expects {} args",
                    arg_names.len()
                )))?;
            }

            mem.push_scope();
            for (name, value) in arg_names.iter().zip(arg_values.iter()) {
                mem.insert_or_modify(name, Rc::clone(value));
            }

            let result = interpret(&body.0, mem);
            mem.pop_scope();
            result
        }
        _ => Err(ValueError("callable is not callable".to_string()))?,
    }
}

fn interpret(exp: &Expression, mem: &mut Mem) -> EvalResult {
    match exp {
        Expression::Number(i) => Ok(Some(Rc::new(Object::Number(*i)))),

        Expression::String(s) => Ok(Some(Rc::new(Object::Str(s.clone())))),

        Expression::Ident(id) => match mem.lookup(&id) {
            Some(value) => Ok(Some(value)),
            None => Err(ValueError(format!("'{}' variable not found", id)).into()),
        },

        Expression::FnDef {
            name,
            arg_names,
            body,
        } => {
            // TODO: clone here is probably can be avoided
            let fn_obj = Object::Function(arg_names.clone(), Ast(*body.clone()));
            mem.insert_or_modify(name, Rc::new(fn_obj));
            Ok(None)
        }

        Expression::ArrayLiteral(v) => {
            let mut results = Vec::new();
            for e in v {
                let r = interpret(e, mem)?.ok_or(ValueError("value expected".to_owned()))?;
                results.push(r);
            }

            Ok(Some(Rc::new(Object::Array(results))))
        }

        Expression::ArrayIndexing { arr_name, expr } => {
            let obj = mem
                .lookup(arr_name)
                .ok_or(ValueError("array not found".to_owned()))?;

            let v = match obj.as_ref() {
                Object::Array(v) => v,
                _ => Err(ValueError("array is expected".into()))?,
            };

            let i = interpret(expr, mem)?.ok_or("index expected")?;

            let i = match *i {
                Object::Number(i) => i,
                _ => Err("integer expected as array index")?,
            };

            if i < 0 {
                Err(format!("index cannot be negative: {}", i).as_str())?
            }

            let i = i as usize;

            if i >= v.len() {
                Err(format!("index out of bounds: {}", i).as_str())?
            }

            Ok(Some(Rc::clone(&v[i])))
        }

        Expression::EqOperator(x, y) => {
            let x = interpret(x, mem)?.ok_or(ValueError("value expected".to_owned()))?;
            let y = interpret(y, mem)?.ok_or(ValueError("value expected".to_owned()))?;

            Ok(Some(Object::Number((x == y) as i64).into()))
        }

        Expression::NotEqOperator(x, y) => {
            let x = interpret(x, mem)?.ok_or(ValueError("value expected".to_owned()))?;
            let y = interpret(y, mem)?.ok_or(ValueError("value expected".to_owned()))?;

            Ok(Some(Object::Number((x != y) as i64).into()))
        }

        Expression::CmpOperator {
            ord_fn: OrdFcn(fcn),
            lhs: x,
            rhs: y,
        } => {
            let x = interpret(x, mem)?.ok_or(ValueError("value expected".to_owned()))?;
            let y = interpret(y, mem)?.ok_or(ValueError("value expected".to_owned()))?;

            // fail comparison for function objects
            use Object::*;
            match (x.as_ref(), y.as_ref()) {
                (Function(_, _), _) | (_, Function(_, _)) => Err("functions cannot be compared")?,
                _ => {}
            }

            let result = fcn(&x, &y);

            Ok(Some(Object::Number(result as i64).into()))
        }

        Expression::Assignment { var_name, expr } => {
            let value = interpret(expr, mem)?;
            match value {
                None => Err(ValueError("value expected".to_owned()).into()),
                Some(value) => {
                    mem.insert_or_modify(&var_name, value);
                    Ok(None)
                }
            }
        }

        Expression::CodeBlock(expr_list) => {
            let mut result = None;
            mem.push_scope();
            for expr in expr_list {
                result = interpret(expr, mem)?;
            }
            mem.pop_scope();
            Ok(result)
        }

        Expression::IfElseBlock {
            cond,
            true_block,
            else_block,
        } => {
            let cond = interpret(cond, mem)?.ok_or(ValueError("value expected".to_owned()))?;

            let is_true = cond.eval_to_bool()?;
            let result = if is_true {
                interpret(true_block, mem)
            } else {
                interpret(else_block, mem)
            };
            result
        }

        Expression::WhileBlock { cond, body } => {
            let mut result = None;
            while {
                let cond = interpret(cond, mem)?.ok_or(ValueError("value expected".to_owned()))?;
                cond.eval_to_bool()?
            } {
                result = interpret(body, mem)?;
            }
            Ok(result)
        }

        Expression::Add(expr_1, expr_2) => {
            eval_args_and_do_ariphmetic(expr_1, expr_2, std::ops::Add::add, mem)
        }

        Expression::Sub(expr_1, expr_2) => {
            eval_args_and_do_ariphmetic(expr_1, expr_2, std::ops::Sub::sub, mem)
        }

        Expression::Mul(expr_1, expr_2) => {
            eval_args_and_do_ariphmetic(expr_1, expr_2, std::ops::Mul::mul, mem)
        }

        Expression::Div(expr_1, expr_2) => {
            eval_args_and_do_ariphmetic(expr_1, expr_2, std::ops::Div::div, mem)
        }

        Expression::Pow(expr_1, expr_2) => {
            eval_args_and_do_ariphmetic(expr_1, expr_2, |x, y| x.pow(y as u32), mem)
        }

        Expression::Neg(expr_1) => {
            eval_args_and_do_ariphmetic(expr_1, &Expression::Number(0), |x, _| x.neg(), mem)
        }

        Expression::FnCall {
            name: fn_name,
            args,
        } => {
            // eval all arg expressions to values
            let mut arg_values = Vec::new();
            for e in args {
                let r = interpret(e, mem)?.ok_or(ValueError("value expected".to_owned()))?;
                arg_values.push(r);
            }

            // lookup fcn name in memory if fails try builtins
            match mem.lookup(fn_name) {
                Some(fcn) => try_call_fn(fcn.as_ref(), &arg_values, mem),
                None => {
                    // try to match builtins
                    let builtin = BUILTINS
                        .get(fn_name)
                        .ok_or(ValueError(format!("no such builtin found: '{}'", fn_name)))?;

                    builtin(&mut stdout(), mem, &arg_values)
                }
            }
        }

        _ => unimplemented!(),
    }
}

fn print_parse_error(err: &peg::error::ParseError<peg::str::LineCol>, src: &str) {
    let offset = err.location.offset;
    use std::cmp::{max, min};

    let mut src_end = offset;

    let failed_char = if offset >= src.len() {
        // when err at EOF we need to limit src_end to src len
        src_end = min(offset, max(src.len() - 1, 0));
        "EOF".to_owned()
    } else {
        src.chars().nth(offset).unwrap().to_string()
    };

    println!(
        "Parse error:\n{} <-- expected {}\nfound: '{}'\n[line: {}, column: {}, offset: {}]",
        &src[0..=src_end],
        err.expected,
        failed_char,
        err.location.line,
        err.location.column,
        offset
    );
}

use std::fs;

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

    // TODO: this should probably by replaced by RAII
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
    println!("\nsnake interpreter 🐍 v0.1 💩\n");

    loop {
        print!("> ");
        std::io::stdout().flush().unwrap();

        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();
        line = normalize_newlines(&line);

        if line.is_empty() {
            continue;
        }

        // try line program into ast and then interpret
        match snake_parser::program(&line) {
            Err(err) => print_parse_error(&err, &line),

            // interpret ast
            Ok(expr_list) => {
                for e in expr_list {
                    match interpret(&e, &mut mem) {
                        Err(err) => println!("Error: {}", &err),
                        Ok(result) => {
                            if let Some(result) = result {
                                println!("{}", result);
                            }
                        }
                    }
                }
            }
        }
    }
}

use regex::Regex;

fn normalize_newlines(src: &str) -> String {
    // normalize newlines to '\n'
    let re = Regex::new(r"(?:\r\n|\r)").unwrap();
    re.replace_all(&src, "\n").to_string()
}

#[test]
fn test_parser() {
    let src = "
        fn print_array(a) {
            i = 0
            while (i < len(a)) {
                print(a[i])
                i = i + 1
            }
        }

        print('printing array:')
        print_array([1, 2, 3])
        a = 10
        b = 1

        while a {
            print(a)
            a = a - 1
        }

        fn sum(a, b) {
            a + b
        }

        print('a + b', sum(a, b))

        a = 1
        b = 1
        c = {
            if(a) {
                a
            }
            else {
                b
            }
        }
        print(a, b)
        mem()
        a / b
        a * b
        a - b
        a + b // comment
        {
            d = a
            d
        }
        print(if a { 'yes' } else { 'no' })

        array = [1, 2, 3]
        array[0]
        len(array)

        print([1, {c = 1 + 1 sum(c, c)}, 2, sum(2 + 2, 2)])

        a > b
        b >= b
        a < b
        a <= b
    ";

    let src = normalize_newlines(src);
    // dbg!(&src);

    // try parse program into ast
    match snake_parser::program(&src) {
        Err(err) => {
            print_parse_error(&err, &src);
            panic!();
        }

        // interpret each expr in ast
        Ok(ast) => {
            let mut memory = Mem::new();

            for statement in &ast {
                interpret(statement, &mut memory).unwrap();
            }
            dbg!(memory);
        }
    }
}

fn execute_file(src_name: String) -> Result<(), AnyError> {
    let src = fs::read_to_string(src_name)?;
    let src = normalize_newlines(&src);

    // try parse program into ast
    match snake_parser::program(&src) {
        Err(err) => {
            print_parse_error(&err, &src);
        }

        // interpret each expr in ast
        Ok(ast) => {
            let mut memory = Mem::new();
            for statement in &ast {
                if let Err(e) = interpret(statement, &mut memory) {
                    println!("Error: {}", &e);
                    std::process::exit(1);
                }
            }
        }
    }

    Ok(())
}

fn get_filename_from_args() -> Option<String> {
    use std::env;
    let mut args = env::args();
    if args.len() > 1 {
        let src = args.nth(1).unwrap();
        return Some(src);
    }
    None
}

use crossterm::{
    cursor,
    event::{poll, read, Event, KeyCode},
    execute,
    style::{Color, Print, SetForegroundColor},
    terminal::{
        self, disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
    },
};

use std::time::Duration;

fn test_crossterm() -> Result<(), AnyError> {
    let mut stdout = stdout();
    let mut mem = Mem::new();

    let args = object_vec![2000];

    snk_enter_game_mode(&mut stdout, &mut mem, &args)?;
    loop {
        if let Object::Array(v) = snk_poll_event(&mut stdout, &mut mem, &args)?
            .ok_or("")?
            .as_ref()
        {
            if *v[0] == "key".into() {
                if *v[1] == "q".into() {
                    break;
                }
            }
        };
    }
    snk_leave_game_mode(&mut stdout, &mut mem, &args)?;

    Ok(())
}

fn snk_print_at_pos(stdout: &mut Stdout, _mem: &mut Mem, args: &Vec<Rc<Object>>) -> EvalResult {
    if args.len() < 3 {
        Err("print_at_pos(x, y, 'text', 'color') expects 4 args")?;
    }

    let x = match args[0].as_ref() {
        Object::Number(x) => *x,
        _ => Err("integer expected")?,
    };

    let y = match args[1].as_ref() {
        Object::Number(y) => *y,
        _ => Err("integer expected")?,
    };

    let text = format!("{}", args[2].as_ref());

    let color = match args[3].as_ref() {
        Object::Str(y) => y,
        _ => Err("str expected")?,
    };

    let color = match color.as_str() {
        "green" => Color::Green,
        "dark_green" => Color::DarkGreen,
        "magenta" => Color::Magenta,
        _ => Color::White,
    };

    queue!(
        stdout,
        cursor::MoveTo(x as u16, y as u16),
        SetForegroundColor(color),
        Print(text.as_str())
    )?;
    Ok(None)
}

fn snk_flush(stdout: &mut Stdout, _mem: &mut Mem, _args: &Vec<Rc<Object>>) -> EvalResult {
    stdout.flush()?;
    Ok(None)
}

fn snk_enter_game_mode(stdout: &mut Stdout, _mem: &mut Mem, _args: &Vec<Rc<Object>>) -> EvalResult {
    enable_raw_mode()?;
    execute!(stdout, EnterAlternateScreen, cursor::Hide)?;
    Ok(None)
}

fn snk_leave_game_mode(stdout: &mut Stdout, _mem: &mut Mem, _args: &Vec<Rc<Object>>) -> EvalResult {
    disable_raw_mode()?;
    execute!(stdout, LeaveAlternateScreen, cursor::Show)?;
    Ok(None)
}

fn snk_clear_screen(stdout: &mut Stdout, _mem: &mut Mem, _args: &Vec<Rc<Object>>) -> EvalResult {
    queue!(stdout, terminal::Clear(terminal::ClearType::All))?;
    Ok(None)
}

fn snk_poll_event(_stdout: &mut Stdout, _mem: &mut Mem, args: &Vec<Rc<Object>>) -> EvalResult {
    if args.len() < 1 {
        Err("poll_event(timeout_ms) expects 1 arg")?
    }

    let timeout_ms = match args[0].as_ref() {
        Object::Number(timeout) => *timeout,
        _ => Err("integer expected")?,
    };

    let ev = if poll(Duration::from_millis(timeout_ms as u64))? {
        match read()? {
            Event::Key(event) => match event.code {
                KeyCode::Char(c) => {
                    Object::Array(object_vec!["key", c.to_string().to_lowercase().as_str()])
                }
                _ => Object::Array(object_vec!["unimpl"]),
            },
            Event::Resize(width, height) => {
                Object::Array(object_vec!["resize", width as i64, height as i64])
            }
            _ => Object::Array(object_vec!["unimpl"]),
        }
    } else {
        Object::Array(object_vec!["timeout"])
    };

    Ok(Some(Rc::new(ev)))
}

fn snk_screen_size(_stdout: &mut Stdout, _mem: &mut Mem, _args: &Vec<Rc<Object>>) -> EvalResult {
    let (w, h) = terminal::size()?;
    let result = object_vec![w as i64, h as i64];

    Ok(Some(Rc::new(Object::Array(result))))
}

fn main() {
    // test_crossterm();
    match get_filename_from_args() {
        Some(file_name) => {
            if let Err(e) = execute_file(file_name) {
                println!("\nError: {}", e);
            }
        }
        None => repl(),
    }
}
