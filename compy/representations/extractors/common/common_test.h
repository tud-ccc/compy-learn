// C samples
constexpr char kProgram1[] =
    "int foo() {\n"
    "  return 1;\n"
    "}";
constexpr char kProgram2[] =
    "int max(int a, int b) {\n"
    "  if(a > b) {\n"
    "    return a;\n"
    "  } else {\n"
    "    return b;\n"
    "  }\n"
    "}";
constexpr char kProgram3[] =
    "#include <stdio.h>\n"
    "\n"
    "void foo() {\n"
    "  printf(\"Hello\");\n"
    "}";
constexpr char kProgram4[] =
    "#include \"tempHdr.h\"\n"
    "\n"
    "void foo() {\n"
    "  barbara(1.2, 3.4);\n"
    "}";
constexpr char kProgram5[] =
    "int max(int a, int b) {\n"
    "  if (a > b) {\n"
    "    return a;\n"
    "  } else {\n"
    "    return b;\n"
    "  }\n"
    "}\n"
    "int foo(int x) {\n"
    "  return max(1, x);\n"
    "}";

// LLVM samples
constexpr char kLLVM1[] =
    "define dso_local void @A(i32*) #0 {\n"
    "  %2 = alloca i32*, align 8\n"
    "  %3 = alloca i32, align 4\n"
    "  store i32* %0, i32** %2, align 8\n"
    "  store i32 2, i32* %3, align 4\n"
    "  %4 = load i32, i32* %3, align 4\n"
    "  %5 = load i32*, i32** %2, align 8\n"
    "  %6 = getelementptr inbounds i32, i32* %5, i64 0\n"
    "  store i32 %4, i32* %6, align 4\n"
    "  ret void\n"
    "}\n";
constexpr char kLLVM2[] =
    "define dso_local void @A(i32*) #0 {\n"
    "  %2 = alloca i32*, align 8\n"
    "  %3 = alloca i32, align 4\n"
    "  store i32* %0, i32** %2, align 8\n"
    "}\n";