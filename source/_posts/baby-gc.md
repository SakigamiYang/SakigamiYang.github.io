---
title: 一个简单的垃圾回收代码
date: 2021-07-20 21:37:01
categories: Develop
tags:
- GC
description: 一个简单的垃圾回收代码，揭示了垃圾回收的基本原理。
---

# 再利用

垃圾回收（Garbage Collection，GC）的基本思想是，编程语言通过底层的操作让你感觉内存是无穷无尽的。但我们都知道内存是有限的，所以要达到这一点就需要将不再被使用的内存（至少看起来是这样）回收成未分配状态。在此之中，最重要的就是**安全地**辨认所谓的“不再被使用”，因为如果你的程序能有机会访问到被抹除为随机值的内存块，那将是很危险的事情。

为了识别能否回收，最简单的思路就是判断一个内存块是否还被至少一个引用指向它。基于此观点，“仍被使用”就可以简单地被认为是：

1. 正在被某变量引用的对象是“仍被使用”的。
2. 被其它对象引用的对象是“仍被使用”的。

当然，第二条意味着它是一个递归，即，如果某变量指向对象 A ，且对象 A 引用了对象 B，那么 B 也是“被使用的”，因为 B 可以从 A 那里被找到。

# 标记-清除（Marking and Sweeping）

垃圾回收有很多算法，其中最简单的就是标记-清除算法，该算法由 John McCarthy 发明（此人也发明了 Lisp ）。它的细节是这样的：

1. 从根节点开始，遍历整个对象依赖图。对每一个能到达的对象，设定一个标记 bit 为 true。
2. 遍历后，删除所有未被标记的对象。

可能会有人说准确的垃圾回收需要更复杂的算法，比如至少要分代回收等，不过作为一个示例我想仅仅从最简单的情况开始。

# 对象对儿

我不想为这个垃圾回收示例写一个复杂的虚拟机（Virtual Machine，VM），所以我们采用一个不碰触语法解析、字节码等这些麻烦事情的方法，定义一个仅含两种类型对象的枚举：

```c
typedef enum {
  OBJ_INT,
  OBJ_PAIR
} ObjectType;
```

其中，这个 OBJ_PAIR 可以包含一对儿任意类型，两个 INT ，一个 INT 一个 PAIR ，或者两个 PAIR 都可以。

细心的朋友们（或接触过函数式编程朋友们）可以发现， PAIR 类型其实是一个可以被定制成很多东西的类型，比如 LIST 类型。事实上，一个 LIST 就可以看做是一个递归 PAIR：

```
LIST = (LIST 首项, LIST 剩余项)
LIST 剩余项 = (LIST 剩余项首项, LIST 剩余项的剩余项)
```

等等。所以能够处理 PAIR 类型其实就意味着你有能力处理很多其它类型。

根据这两种类型，我们定义一个结构体，让两种类型共享一个值，从而达到模拟 VM 中一个变量只能对类型进行二选一的特性。

```c
typedef struct sObject {
  ObjectType type;

  union {
    /* OBJ_INT */
    int value;

    /* OBJ_PAIR */
    struct {
      struct sObject* head;
      struct sObject* tail;
    };
  };
} Object;
```

# 一个小型 VM

有了定义对象的方法，我们就可以做一个小型 VM 了，这个 VM 只包含一个有限长的对象栈。

```c
#define STACK_MAX 256

typedef struct {
  Object* stack[STACK_MAX];
  int stackSize;
} VM;
```

紧接着，我们需要定义 new 函数、push 和 pop 函数，这些都是基本操作。

```c
VM* newVM() {
  VM* vm = malloc(sizeof(VM));
  vm->stackSize = 0;
  return vm;
}

void push(VM* vm, Object* value) {
  assert(vm->stackSize < STACK_MAX, "Stack overflow!");
  vm->stack[vm->stackSize++] = value;
}

Object* pop(VM* vm) {
  assert(vm->stackSize > 0, "Stack underflow!");
  return vm->stack[--vm->stackSize];
}
```

定义如何在 VM 里分配空间给一个对象（“变量”）。

```c
Object* newObject(VM* vm, ObjectType type) {
  Object* object = malloc(sizeof(Object));
  object->type = type;
  return object;
}
```

注意到 PAIR 类型是一种复合对象，所以我们采用比较栈风格的方法来创造 PAIR 对象，即 PAIR 对象是由弹出栈顶两个对象再怼进来一个新 PAIR 对象做的。

```c
void pushInt(VM* vm, int intValue) {
  Object* object = newObject(vm, OBJ_INT);
  object->value = intValue;
  push(vm, object);
}

Object* pushPair(VM* vm) {
  Object* object = newObject(vm, OBJ_PAIR);
  object->tail = pop(vm);
  object->head = pop(vm);

  push(vm, object);
  return object;
}
```

到此为止，VM 就大功告成了。

# 标记

为了标记，我们需要在对象中放置一个标志位。

```c
typedef struct sObject {
  unsigned char marked;
  /* Previous stuff... */
} Object;
```

而刚刚所说的标记-清理的第一步，就是遍历 VM 中栈的所有对象并进行标记。注意，PAIR 是个复合对象，如果一个 PAIR 对象被标记了，其引用的对象也应该被标记。

```c
void mark(Object* object) {
  object->marked = 1;

  if (object->type == OBJ_PAIR) {
    mark(object->head);
    mark(object->tail);
  }
}

void markAll(VM* vm)
{
  for (int i = 0; i < vm->stackSize; i++) {
    mark(vm->stack[i]);
  }
}
```

但是这个代码有个问题，如果对象的引用图上有圈，那么就会造成死循环，所以我们改一下 mark 函数为：

```c
void mark(Object* object) {
  /* If already marked, we're done. Check this first
     to avoid recursing on cycles in the object graph. */
  if (object->marked) return;

  object->marked = 1;

  if (object->type == OBJ_PAIR) {
    mark(object->head);
    mark(object->tail);
  }
}
```

# 清理

标记-清理的第二步就是清理，这里一上来就有一个问题，即按照定义，未标记的对象应该是不可达的，那既然是不可达的我们也没法知道它在哪儿，从而无法回收它。

所以在这里我们就需要 C 语言里最强也是最恶心的东西了——指针。

我们用一个小技巧就是在每一个对象中记录下一个被分配的对象的指针，同时在 VM 里记住第一个被分配的对象是谁：

```c
typedef struct sObject {
  /* The next object in the list of all objects. */
  struct sObject* next;

  /* Previous stuff... */
} Object;

typedef struct {
  /* The first object in the list of all objects. */
  Object* firstObject;

  /* Previous stuff... */
} VM;
```

同时我们稍稍修改 VM 的 new 函数，初始化时让 firstObject 指向 NULL。

```c
Object* newObject(VM* vm, ObjectType type) {
  Object* object = malloc(sizeof(Object));
  object->type = type;
  object->marked = 0;

  /* Insert it into the list of allocated objects. */
  object->next = vm->firstObject;
  vm->firstObject = object;

  return object;
}
```

这样做我们就可以用指针记住要被删除的对象，还记得 Linux 内核里链表结构中指针的指针这一技巧吗？

```c
void sweep(VM* vm)
{
  Object** object = &vm->firstObject;
  while (*object) {
    if (!(*object)->marked) {
      /* This object wasn't reached, so remove it from the list
         and free it. */
      Object* unreached = *object;

      *object = unreached->next;
      free(unreached);
    } else {
      /* This object was reached, so unmark it (for the next GC)
         and move on to the next. */
      (*object)->marked = 0;
      object = &(*object)->next;
    }
  }
}
```

现在我们可以完成垃圾回收了，先标记再清除即可：

```c
void gc(VM* vm) {
  markAll(vm);
  sweep(vm);
}
```

# 更多的问题

怎么判断 VM 低内存呢？如果计算垃圾回收的时点？

这些问题没有特别固定的解决办法，在我的例子里，添加一个对对象的计数并给定一个垃圾回收底线即可：

```c
typedef struct {
  /* The total number of currently allocated objects. */
  int numObjects;

  /* The number of objects required to trigger a GC. */
  int maxObjects;

  /* Previous stuff... */
} VM;

--------------------

VM* newVM() {
  /* Previous stuff... */

  vm->numObjects = 0;
  vm->maxObjects = INITIAL_GC_THRESHOLD;
  return vm;
}

--------------------

Object* newObject(VM* vm, ObjectType type) {
  if (vm->numObjects == vm->maxObjects) gc(vm);

  /* Create object... */

  vm->numObjects++;
  return object;
}

--------------------

void gc(VM* vm) {
  int numObjects = vm->numObjects;

  markAll(vm);
  sweep(vm);

  vm->maxObjects = vm->numObjects * 2;
}
```

简单地说，我用一个不断扩大成 2 倍的数字来记录，比如：第一次有 8 个对象就垃圾回收；第二次的话，我就在判断“系统中很轻易的就能达到 8 个对象”的前提下，16 个对象再做回收。以此类推。

# 最后

这样就 OK 了，我们完成了一个简单的垃圾回收模块。它很简单，但是足以说明问题。

全部的代码请参考我的 GitHub：[SakigamiYang/baby-gc](https://github.com/SakigamiYang/baby-gc)

