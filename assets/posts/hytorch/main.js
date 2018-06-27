tree.render_ast(
  '#fib-prog-container', 800, 900,
  {
    name: 'fib.c',
    children: [
      {
        name: 'int func fib',
        children: [
          { name: 'int param n' },
          {
            name: 'int var a',
            children: [ { name: '=', children: [ { name: '1' } ] } ],
          },
          {
            name: 'int var b',
            children: [ { name: '=', children: [ { name: '1' } ] } ],
          },
          {
            name: 'int var t',
          },
          {
            name: 'int var i',
          },
          {
            name: 'for',
            children: [
              {
                name: 'i',
                children: [ { name: '=', children: [ { name: '2' } ] } ],
              },
              {
                name: 'if',
                children: [ { name: '<' }, { name: 'i' }, { name: 'n' } ],
              },
              {
                name: '++',
                children: [ { name: 'i' } ],
              },
              {
                name: 'do',
                children: [
                  {
                    name: 't',
                    children: [ { name: '=', children: [ { name: 'a' } ] } ],
                  },
                  {
                    name: 'a',
                    children: [ { name: '=', children: [ { name: 'b' } ] } ],
                  },
                  {
                    name: 'b',
                    children: [
                      {
                        name: '=',
                        children: [
                          {
                            name: '+',
                            children: [ { name: 't' }, { name: 'b' } ],
                          },
                        ],
                      },
                    ],
                  },
                ],
              },
            ],
          },
          {
            name: 'return',
            children: [ { name: 'b' } ],
          },
        ],
      },
      {
        name: 'void func main',
        children: [
          {
            name: 'printf',
            children: [
              {
                name: '"%d\n"',
              },
              {
                name: 'fib',
                children: [ { name: '12' } ],
              },
            ],
          },
        ],
      },
    ],
  },
);

tree.render_ast(
  '#fib-naive-scheme-prog-container', 800, 400,
  {
    name: 'fib.scm',
    children: [
      {
        name: 'func fib',
        children: [
          {
            name: 'param n',
          },
          {
            name: 'if',
            children: [
              {
                name: '<',
                children: [ { name: 'n' }, { name: '2' } ],
              },
              {
                name: 'n',
              },
              {
                name: '+',
                children: [
                  {
                    name: 'fib',
                    children: [
                      {
                        name: '-',
                        children: [ { name: 'n' }, { name: '1' } ],
                      },
                    ],
                  },
                  {
                    name: 'fib',
                    children: [
                      {
                        name: '-',
                        children: [ { name: 'n' }, { name: '2' } ],
                      },
                    ],
                  },
                ],
              },
            ],
          },
        ],
      },
      {
        name: 'display',
        children: [ { name: 'fib', children: [ { name: '12' } ] } ],
      },
    ],
  },
);

tree.render_ast(
  '#fib-better-scheme-prog-container', 800, 500,
  {
    name: 'fib.scm',
    children: [
      {
        name: 'func fib',
        children: [
          {
            name: 'param n',
          },
          {
            name: 'func fib-aux',
            children: [
              { name: 'param n' },
              { name: 'param a' },
              { name: 'param b' },
              {
                name: 'if',
                children: [
                  {
                    name: '=',
                    children: [ { name: 'n' }, { name: '2' } ],
                  },
                  { name: 'a' },
                  {
                    name: 'fib-aux',
                    children: [
                      {
                        name: '-',
                        children: [ { name: 'n' }, { name: '1' } ],
                      },
                      {
                        name: 'b',
                      },
                      {
                        name: '+',
                        children: [ { name: 'a' }, { name: 'b' } ],
                      },
                    ],
                  },
                ],
              },
            ],
          },
          {
            name: 'fib-aux',
            children: [ { name: 'n' }, { name: '0' }, { name: '1' } ],
          },
        ],
      },
      {
        name: 'display',
        children: [ { name: 'fib', children: [ { name: '12' } ] } ],
      },
    ],
  },
);
