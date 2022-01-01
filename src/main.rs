use::lazy_static::lazy_static;
use::std::collections::HashMap;
use std::{result, io::{stdout, Write}};

#[derive(Debug)]
pub enum Expression {
    Void,
    Ident(String),
    Number(i64),
    String(String),
    Assignment(String, Box<Expression>),
    Add(Box<Expression>, Box<Expression>),
    FnCall(String, Box<Expression>)
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

peg::parser!( grammar test_parser() for str {
    rule _ = [' ']*
    rule br() = ['\n']*

    rule plus() = "+"

    pub rule ident() -> &'input str
        = s: $(['a'..='z' | 'A'..='Z']['_' | 'a'..='z' | 'A'..='Z' | '0'..='9']*) { 
            s
        }
    
    rule number() -> Expression
        = n: $(['0'..='9']+) { Expression::Number(n.parse().unwrap())}

    pub rule expression() -> Expression = precedence! {
        x:(@) plus() y:@ { 
            Expression::Add(x.into(), y.into()) 
        }

        _ f:ident() "(" e:expression() ")" _ { Expression::FnCall(f.to_owned(), e.into())} 

        --
        _ i:ident() _ "=" _ n:expression() { Expression::Assignment(i.to_string(), n.into()) }
        _ i:ident() _ { Expression::Ident(i.to_string()) }
        _ n:number() _ { n }
        _ "'" t:$(['a'..='z' | 'A'..='Z' | ' ']+) "'" _ { Expression::String(t.to_owned()) }
        _ "" _ { Expression::Void }
    }

    pub rule program() -> Vec<Expression>
        = br() _ s:expression() ** "\n" br() _ { s }
});

fn interpret(exp: Expression, mem: &mut Mem) -> Option<String> {
   match exp {
       Expression::Number(i) => {
           Some(i.to_string())
       },

       Expression::String(s) => {
           Some(s)
       }

       Expression::Ident(id) => {
           let value = mem.get(&id).expect(format!("'{}' variable not found", id).as_str());
           Some(value.clone())
       }

       Expression::Assignment(var_name, expr) => {
            let value = interpret(*expr, mem).unwrap();
            mem.insert(var_name, value);
            None
       },

       Expression::Add(expr_1, expr_2) => {
           let value_1 = interpret(*expr_1, mem).expect("value expected");
           let value_2 = interpret(*expr_2, mem).expect("value expected");

           Some(
               (value_1.parse::<i64>().unwrap() + value_2.parse::<i64>().unwrap())
               .to_string()
           )
       }

       Expression::FnCall(fn_name, expr) => {
            let value = interpret(*expr, mem).expect("value expected");

            let builtin = BUILTINS.get(&fn_name).expect("no such builtin found");
            builtin(value);
            None
       }
       _ => unimplemented!()
   } 
}

type Mem = std::collections::HashMap::<String, String>;

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

        let result = std::panic::catch_unwind(|| {
            let expr = test_parser::program(&line).expect("failed to parse").remove(0);
            expr
        });

        if let Err(_) = result {
            continue;
        }

        let expr = result.unwrap();

        if let Some(result) = interpret(expr, mem) {
            println!("{}", result);
        }
    }
}

fn main() {
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

    let ast = test_parser::program(src).expect("failed to parse");

    let mut memory = Mem::new();

    repl(&mut memory);

    for statement in ast {
        interpret(statement, &mut memory);
    }

    dbg!(memory);
}

