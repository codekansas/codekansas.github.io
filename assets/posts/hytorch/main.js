tree.render_ast(
  '#simple-prog-container', 800, 600,
  {
    name: 'simple.c',
    children: [
      {
        name: 'int var x',
        children: [
          {
            name: '5',
          },
        ],
      },
      {
        name: 'int var y',
        children: [
          {
            name: '3',
          },
        ],
      },
      {
        name: 'int func my_function',
        children: [
          {
            name: 'int param a',
          },
          {
            name: 'int param b',
          },
          {
            name: 'int var c',
            children: [
              {
                name: '+',
                children: [
                  {
                    name: 'a',
                  },
                  {
                    name: 'b',
                  },
                  {
                    name: 'y',
                  },
                ],
              },
            ],
          },
          {
            name: 'printf',
            children: [
              {
                name: '"c: %d\n"',
              },
              {
                name: 'c',
              },
            ],
          },
          {
            name: 'return',
            children: [
              {
                name: 'c',
              },
            ],
          },
        ],
      },
      {
        name: 'int func main',
        children: [
          {
            name: 'int var d',
            children: [
              {
                name: 'my_function',
                children: [
                  {
                    name: '5',
                  },
                  {
                    name: 'x',
                  },
                ],
              }
            ],
          },
          {
            name: 'printf',
            children: [
              {
                name: '"d: %d\n"',
              },
              {
                name: 'd',
              }
            ],
          },
          {
            name: 'return',
            children: [
              {
                'name': '0',
              },
            ],
          },
        ],
      },
    ],
  },
);
