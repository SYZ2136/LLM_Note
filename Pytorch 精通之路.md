
# 变量命名规范
```
类名: MultiHeadAttention (每个单词首字母都大写)
函数/变量：snake_case
命令行参数传入: --kebab-case 
e.g. 
python train_sdd.py --train data/train.csv --val data/val.csv --epochs 100 --batch-size 128 --lr 1e-3 --num-shapelets 20 --shapelet-length 20 --patch-len 5 --save-dir checkpoints_len20_n5
其中每个变量之前的 -- 是命令行选项前缀，单词之间通常用一个 - 分隔是 kebab-case 的惯例，parser.add_argument("--batch-size") 解析后就是 args.batch_size
```


---



# 类与对象
`对象(object)`是`类(class)`的`实例(instance)`，`class` 本身也是 `type`类的实例
Python 中一切皆对象，都能用 `type()` 得到它的类型，`return` 可以返回任何 Python 对象
 
 `return` 返回的常见类别包括：
- **基础类型**：`int / float / bool / str / bytes`
- **容器**：`list / tuple / dict / set`
- **类本身**：`return MyClass`
- **类实例化出的对象**
- **函数/方法本身**（函数也是对象）：`return some_function`
- **`PyTorch` / `NumPy` 对象**：`torch.Tensor`、`nn.Module`、`np.ndarray`、`np.float64` 等


---


# 逻辑运算优先级

不额外加括号的情况下，优先级有
```
not  >  and  >  or
```

---

# and 和 or 是短路求值


`A or B`, 如果 A 真，B 直接就不看了，压根不会读取执行
`A and B`, 如果 A 为 假， 同样 B 直接不看了

```python
if start == len(nums) or nums[start] != target:  # 力扣 34
# 这里的两个条件不能换序，因为前一个条件为真后半部分不看不执行，这样就避免了数组索引超出的报错，只有前面为假，才读后面，此时后面条件也合法了
```



---



# 多目标赋值

`nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]`
和
`nums[i], nums[nums[i] - 1] = nums[nums[i] - 1], nums[i]
不等价

多目标赋值的语义为：
     先计算右边所有表达式，得到一个临时的“值元组”
     左边的式子从左到右依次赋值


上面式子 `nums[nums[i] - 1]` 依赖于 `nums[i]`, 
如果写成 `nums[i], nums[nums[i] - 1] = nums[nums[i] - 1], nums[i]`，那么`nums[i]` 直接先发生改变了

```
x = 1, 2, 3  # x 打印出来就是 (1, 2, 3), 类型为元组

# 下面的迭代也是，右侧的四个二元组整体这样逗号式子本身就会自动打包成元组，所有不用额外外侧再加一个括号
for i, j in (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1):
```


---


# self 和 实例属性

#### self

**self就是一个未来实例的“代称”，实例化之后，self就是实例对象本身**，
self 服务于实例属性和实例方法的东西 (有类属性和实例属性两种东西)
```python
class SimpleNet(nn.Module): 
    class_var = "类属性"    # 这个就是类属性  
    def __init__(self):    # 这个定义的是实例属性  
        super().__init__()  
        self.fc1 = nn.Linear(10, 5)  
    def print_self_info(self):  
        print(f"self的类型: {type(self)}")  
        print(f"self的id: {id(self)}")  
        print(f"self就是实例本身: {self is model}")    # 返回True

# 创建实例
model = SimpleNet()  
print(f"model的id: {id(model)}")  
# 调用方法
model.print_self_info()
# model的id: 1845422618992
# self的类型: <class '__main__.SimpleNet'>
# self的id: 1845422618992
# self就是实例本身: True
```

确实验证了：实例化之后，self就是实例对象本身；
并且，类的定义的过程是一个蓝图，在这个过程中，既然self是未来实例对象的代称，那么在蓝图里面self.method()自然是可行的，因为反正是蓝图，真调用说明已经实例化了，那都已经实例化了self就是实例化对象，当然能够调用相应方法


#### 类的属性

实例属性定义之后，实例化之后仍然可以被修改、添加
```
class Counter:
    def __init__(self):
        self.count = 0   # 给当前对象一个属性 count

    def add_one(self):
        self.count = self.count + 1
        
c = Counter()        
c.count  # 0
c.add_one()
c.count  # 1

class Person:
    def __init__(self, name):
        self.name = name

    def set_age(self, age):
        self.age = age
a = Person("Tom")
a.__dict__  # {'name': 'Tom'}
a.set_age(18)
a.__dict__  # {'name': 'Tom', 'age': 18}
```


----


# super()

super本质上是一个类，super()是一个类的调用，
```python
class Animal:
    def __init__(self, age):
        self.age = age


class Person(Animal):
    def __init__(self, age, name):
        super().__init__(age)
        self.name = name


class Male(Person):
    def __init__(self, age, name):
        super().__init__(age, name)
        self.gender = "male"
        
m = Male(18, "Tom")
```

`super().__init__(age, name)` 理解成 `Person.__init__(self, name)`
这样本身 Male 的 `__init__` 的两个参数，传入父类的 `__init__`
写成 super 而非 Person. 的优势是可以动态修改，当父类子类一些逻辑改的时候能实现自动匹配。



---


# 装饰器

函数装饰器本质上就是**一个接收函数作为参数，并返回一个新函数的函数**
```
# 这里定义一个装饰器，比如传入 hello() 函数，实现头尾多打印两句
def decorator(func):
    def wrapper():
        print("开始执行")
        func()
        print("执行结束")
    return wrapper

def hello():
    print("hello")
hello = decorator(hello)
hello()
```
python 提供更简洁的写法：`@装饰器` 紧贴着写在目标函数上一行；所以装饰器在目标函数定义时就执行了；比如这里 `def hello` 时上面加个装饰器，就等价于 `hello = decorator(hello)`
```
def decorator(func):
    def wrapper():
        print("开始执行")
        func()
        print("执行结束")
    return wrapper

@decorator
def hello():
    print("hello")
hello()
```

#### `@cache`
本质上就是一个字典，修饰的目标函数的参数作为键，返回值作为值；
因为修饰函数的参数作为键写入字典，**所以修饰的函数必须是可哈希的**，`list,dict,set` 这类就不能作为函数参数，多组参数组合成 `tuple` 作为键

第一次碰到，看看字典键里有没有这一组参数，如果没有，就执行函数，然后存入字典，如果有，就直接查找字典，得到这组参数值下的函数返回值

#### `@staticmethod`

普通实例方法不传入 self 的情况：
```
class MathTool:
    def add(a, b):
        return a + b
```
此时`MathTool.add(1, 2)` 合法;
`m = MathTool()` 然后 `m.add(1, 2)`  不合法，因为 Python 会自动把 m 当成第一个参数传进去，等价于 `MathTool.add(m, 1, 2)`，定义的时候写了两个参数，传入三个参数，自然不合法

写成静态方法就合法了：
```
class MathTool:
    @staticmethod
    def add(a, b):
        return a + b
```
加了 `@staticmethod` 不会自动绑定 `self`,此时`MathTool.add(1, 2)` 和 `m = MathTool()` 然后 `m.add(1, 2)` 都合法


# set 语法

`set` 是 Python 的一种内置容器类型，表示“**不重复元素的集合**”
`set` 是一个“只存 key 的哈希表结构”，只关心元素是否存在。
集合 set 是一个特殊的字典（只存 key，不存 value），其底层实现是一个哈希表

```
s = {1, 2, 3}  # 创建set

s = set([1, 2, 9， 9, 3])
print(s)  # {1, 2, 3, 9}  自动去重

# 有些 IDE/Notebook）为了输出稳定好看，会偷偷sorted(set),这样看似打印出的结果就是排序好的

for x in s: 
    print(x)  # 可验证实际顺序

s = set()   # 空set只能这么创建
s = {}      # ❌ 这是空字典

for _ in set  # 迭代顺序由hash表内部布局决定
```
###### 特性
- **无序**（不能靠下标取值）
- **元素唯一**（自动去重）
- **元素必须可哈希**（一般是不可变类型，如 int/str/tuple；list/dict 不能放进去）

###### 增删改查
```
s = {1, 2}

s.add(3)  # 增加一个元素, 如果添加的元素已经在 set 里，就什么都不发生
print(s)  # {1, 2, 3}

# 批量加入用update
s = {1, 2}
s.update([2, 3, 4])
print(s)  # {1, 2, 3, 4}

# 删除
s = {1, 2, 3}
s.remove(2)   # 原地操作，返回值为None，{1, 3}
s.discard(5)  # 原地操作，返回值为None,不存在也不会报错
t = s - {1}  # t 为{2, 3}, s 还是 {1, 2, 3}
# python 中大部分原地操作的方法返回值都是 None，目的就是提示这个方法是原地修改原对象，而不是生成一个新对象

# 判断是否在集合里，很快，平均 O(1)
s = {1, 2, 3}
print(2 in s)      # True 
print(5 not in s)  # True

# 以下这些集合运算都是非原地，原集合 a,b 本身都不变，返回新集合
a = {1, 2, 3}
b = {3, 4, 5}
# 并集, 以下每组中两种写法等价
a | b
a.union(b)  # 非原地操作，a.union(b) == b.union(a) 
# 交集
a & b
a.intersection(b)
# 差集
a - b
a.difference(b)
# 对称差(只在其中之一)
a ^ b
a.symmetric_difference(b)
```

# list 语法


增删改查
```
# 增
lst = [1, 2]
lst.append(3)      # append() 是尾部添加一个元素， [1, 2, 3]
lst.append([4, 5]) # [1, 2, 3, [4, 5]]

lst = [1, 2]  # extend() 把一个可迭代对象一个个加到尾部
lst.extend([3, 4])   # [1, 2, 3, 4]
lst.extend("ab")     # [1, 2, 3, 4, 'a', 'b']
d = {"a": 10, "b": 20}
lst.extend(d)  # 字典默认迭代键

a = [1, 2]
b = a
# a += b → 调用 a.__iadd__(b)，原地加
# a + b → 调用 a.__add__(b)，普通加，产生新对象
a += [3, 4]  # += 是原地操作，id(a) == id(b)，变为 [1, 2, 3, 4]，相当于 .extend()
a = a + [6 ,7]  # 直接加再赋值是非原地，此时 id(a) ≠ id(b)

# 删
lst = [1, 2, 3, 4]
a = lst.pop()  # 默认删除最后一个， a 结果为 4， lst 变为 [1, 2, 3]
lst = [5, 6, 7, 8]
b = lst.pop(2)  # 删除索引为 2 的元素

print(a)    # 3
print(lst)  # [1, 2, 4]


print(a)    # 3
print(lst)  # [1, 2, 4]



# 拷贝
list.copy()  # 无参数，拷贝一份副本； list， set， dict 都支持 .copy() 方法



s = "ab" * 3        # "ababab"
t = (0,) * 4        # (0, 0, 0, 0), "(0)"只是一个加了括号的表达式，再加一个comma才能被识别为元组
b = [0] * 10        # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


# 返回 list 中某个数值的索引
list = [10, 20, 30, 40]
list.index(30)  # 2
```

list 中的元素可以是任意类型，也可以是自己定义的类
```
智谱 API 调用 response 返回的一个东西
stream 参数为 false 时，response 结果为一个完整的 completion；如果stream=True → response 就是可迭代对象，每次迭代为一个增量片段chunk

Completion(model='glm-4.6v', created=1772270990, choices=[CompletionChoice(index=0, finish_reason='stop', message=CompletionMessage(content='\n这张图片展示的是一串香蕉（一串成熟的黄色香蕉，部分香蕉带有贴纸）。', role='assistant', reasoning_content='用户现在需要判断这张图片是什么。首先看图片内容，是一串香蕉，黄色的，上面还有贴纸。所以要确认这是香蕉，一串成熟的香蕉，可能带有品牌贴纸。所以回答应该是这串水果是香蕉，具体来说是一串黄色的香蕉，可能带有标签。现在组织语言：这张图片展示的是一串黄色的香蕉，香蕉表面光滑，颜色鲜亮，其中一根香蕉上贴有标签，整体看起来是成熟的新鲜香蕉。', tool_calls=None))], request_id='20260228172943c6096e9117f74967', id='20260228172943c6096e9117f74967', usage=CompletionUsage(prompt_tokens=75, completion_tokens=124, total_tokens=199, completion_tokens_details={'reasoning_tokens': 97}, prompt_tokens_details={'cached_tokens': 4}), object='chat.completion')

response.choices 为一个单元素 list，这个元素为类 CompletionChoice
[CompletionChoice(index=0, finish_reason='stop', message=CompletionMessage(content='\n这张图片展示的是一串香蕉（一串成熟的黄色香蕉，部分香蕉带有贴纸）。', role='assistant', reasoning_content='用户现在需要判断这张图片是什么。首先看图片内容，是一串香蕉，黄色的，上面还有贴纸。所以要确认这是香蕉，一串成熟的香蕉，可能带有品牌贴纸。所以回答应该是这串水果是香蕉，具体来说是一串黄色的香蕉，可能带有标签。现在组织语言：这张图片展示的是一串黄色的香蕉，香蕉表面光滑，颜色鲜亮，其中一根香蕉上贴有标签，整体看起来是成熟的新鲜香蕉。', tool_calls=None))]

最终 response.choices[0].message.content 的结果为
'\n这张图片展示的是一串香蕉（一串成熟的黄色香蕉，部分香蕉带有贴纸）。'

```



---



# 推导式

### list 推导式
简洁创建列表的方法，代替多行循环语句
```
squares = [x**2 for x in range(10)]
# 结果为 [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

results = [x if x % 2 == 0 else 'odd' for x in range(5)]
# 结果为 [0, 'odd', 2, 'odd', 4]

coordinates = [(x, y) for x in range(2) for y in range(3)]
# 结果为 [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

visited = [[False] * 3 for _ in range(2)]
# [[False, False, False], [False, False, False]]
```

### dict 推导式
```
squares = {x: x**2 for x in range(3)}
# 结果为 {0: 0, 1: 1, 2: 4}

a = ['apple', 'banana', 'cherry']
b = {x: len(x) for x in a}
# 结果为 {'apple': 5, 'banana': 6, 'cherry': 6}
```


---

# "".join

`str.join(iterable)` 表示用 str 作为分隔符，把可迭代对象中的每个元素连起来
只有字符串对象可以 `.join`，且传入的对象也只能是字符串

```
"12".join("abc")  # 结果为'a12b12c'

# 第一种，for 的迭代器语法
''.join(['1', '2', '3'])  # "123"
c = ''.join(str(x) for x in a)

# 第二种，惰性迭代器对象 map；注意 reversed(a) 返回的是一个 list_reverseiterator 对象
a = [1, 2, 3]
c = "".join(map(str, reversed(a)))  # '321'
```



---


# reversed()

惰性迭代器
`reversed(dict)` 反的是字典创建时的插入顺序
```
d = {'a': 1, 'b': 2, 'c': 3}
print(list(d))           # ['a', 'b', 'c']
print(list(reversed(d))) # ['c', 'b', 'a']


buckets = {}  
buckets[2] = ['a']  
buckets[1] = ['b']  
buckets[3] = ['c']
list(reversed(buckets))  # [3, 1, 2]
```

---


# map()

`map(func, iterable)` 是一个内置函数，返回一个惰性迭代器 **map 对象**，每次迭代 map 对象得到 `func(x)`

```
nums = ["1", "2", "3"]
a = list(map(int, nums))  # [1, 2, 3]
type(map(int, nums))  # map
# map 是一个惰性迭代器，被迭代时才逐个“发出”元素。
```

```
nums = ["1", "2", "3"]
for x in map(int, nums):
    print(type(x))
# 结果为 <class 'int'> <class 'int'> <class 'int'>
```



---



# 切片

### list 切片语法
```
a = [0,1,2,3,4,5,6,7,8,9]

# 头或尾省略表示包含，a[6:] 包含尾，a[:4] 包含头，a[:] 全序列头尾都包含
a[2:6]      # 左闭右开，a[2]到a[6]，这种切片都是拷贝
a[:4]       # 包含开头到索引4，右开
a[6:]       # 左闭，到结尾，包含结尾
a[:]        # [0,1,2,3,4,5,6,7,8,9]  (拷贝，对 list 来说是新列表)

a[1:9:2]    # 左闭右开，步长为2，[1,3,5,7]
a[8:3:-1]    # 左闭右开，步长为-1, [8, 7, 6, 5, 4]
a[::-1]    # 两头都省略，开头结尾都包含，就是一个逆转的拷贝

a[-3:]      # 倒数第三个到最后，包含结尾[7,8,9] 
a[:-3]      # [0,1,2,3,4,5,6]

# 冒号两边都有索引，就是左闭右开，省略的话，开头或者结尾包含在内

a[:] 在右边：表示“取整个列表的一个切片副本”（会创建新列表）。
a[:] 在左边：表示“把这个切片对应的那段位置，用右侧序列替换掉”（原地替换）。

a[:] = a[::-1]  # 可以实现 a 原地逆序，但是右侧仍然会新建一个逆序的拷贝，空间O(n)
a.reverse()  # 原地逆序，时间 O(n),空间 O(1)，很高效

a = [1, 2, 3]
b= a[:]  
a is b  # False

a = [1, 2, 3]
b = a            # b 和 a 指向同一个列表对象
a[:] = [10, 20]
a is b  # True, 二者都为[10, 20]

nums[:] = nums[-k:] + nums[:-k]  # 数组进行平移，原地操作(见 leetcode 189)
```

### str, tuple, list 都是 sequence 类型

三者都支持切片索引语法，语法规则一样



---




# deque 语法
queue 是先进先出，队尾入队，队首出队
deque 队首队尾都可以入队出队
```
from collections import deque

d = deque()
d = deque([1, 2, 3])

# 尾部追加（右边）
d.append(1)
d.append(2)       # d -> [1, 2]
# 尾部弹出（右边）
x = d.pop()       # x = 2, d -> [1]

# 头部入队出队
d = deque([1, 2])
d.appendleft(0)   # d -> [0, 1, 2]
x = d.popleft()   # x = 0, d -> [1, 2]

# 批量扩展
d = deque([1])
d.extend([2, 3])      # d -> [1, 2, 3]
# 依次从左插入，所以是“反着”插入
d.extendleft([0, -1]) # d -> [-1, 0, 1, 2, 3]

# 
d = deque([1, 2, 3])
len(d)        # 3
# 下标索引，不是 O(1)
first = d[0]      # 1
last = d[-1]      # 3
1 in d       # True
```



---


# Counter()

`Counter` 本质上是一个**计数器字典**
`Counter` 本质上是 `dict` 的子类，它的迭代顺序就是键值对的插入顺序。

```
cnt = Counter()
cnt["a"]  # 查找不存在的键的时候，返回 0，但是 cnt 本身不会加这个键值对，这里还是 Counter()

cnt["a"] += 1  # 这样有次数了，cnt 就变成了Counter({'a': 1})

cnt = Counter(iterable)  # 比如 list，tuple，dict, str 等
cnt = Counter(["a", 1, 2, 3, 3, 3])  # Counter({3: 3, 'a': 1, 1: 1, 2: 1})
cnt = Counter(a=3, b=2)  # 按关键字传入，键当成字符串，Counter({'a': 3, 'b': 2})

# 传入字典是直接 mapping 成对应字典，而不是迭代键
cnt = Counter({'a': 3, 'b': 2})  # Counter({'a': 3, 'b': 2})


# 元素高频到低频
cnt = Counter("abacaba")
c = cnt.most_common()  # [('a', 4), ('b', 2), ('c', 1)]，即返回一个 list
# 取前 k 个高频元素
cnt = Counter("abacaba")
c = cnt.most_common(2)  #返回 [('a', 4), ('b', 2)] 这个 list


# 加减统计元素
cnt = Counter("aab")
cnt.update("bcc")  # cnt 变为 Counter({'a': 2, 'b': 2, 'c': 2})
cnt = Counter("aabcc")
cnt.subtract("abc")  # cnt 变为 Counter({'a': 1, 'c': 1, 'b': 0})

# Counter 计数过程无排序
nums = [5, 5, 2, 2, 2, 9]
cnt = Counter(nums)  # Counter({2: 3, 5: 2, 9: 1})，打印出来只是看上去次数总是降序
list(cnt.items())    # [(5, 2), (2, 3), (9, 1)]，真实迭代是按照字典创建的插入顺序
```



---



# dict 和 defaultdict 语法

#### dict
`dict` 的 `key` 必须是 hashable 的对象，不可变对象 `list,dict,set` 不可哈希，不能作为字典的键

增删改查
```
# 有则改，无则新增
d = {}
d["a"] = 1      # 新增 key 'a'
d["a"] = 10     # 修改 key 'a' 对应的值

# 创建，以下三种方式等价
d = {"a": 1, "b": 2}
d = dict(a=1, b=2)  # 只适用于 key 为字符串且变量名合法的情况

# update
d = {"a": 1}
d.update({"b": 2, "a": 10})  # -> {"a": 10, "b": 2}
d.update(c=3, d=4)  # -> {"a": 10, "b": 2, "c": 3, "d": 4}

# 判断键或值在不在字典里
x in d  # 查键，O(1)
x in d.values()  # 查值，O(n),因为值没有建立哈希表索引，要一个一个找
```
`dict.get()` 是 `dict` 的内置方法，如果键存在，就返回键值，如果键不存在，则返回 default
```
dic.get(key, default)  # default 默认值为 None

counts = {"a": 2, "b": 5}

counts.get("a", 0)  # 返回 2 
counts.get("c", 0)  # 返回 default=0
```


#### defaultdict
`defaultdict` 是一个带默认值的特殊的 `dict` 
```
from collections import defaultdict
d = defaultdict(int)     # 默认值是 0
d = defaultdict(set)     # 默认值是 {}

# 当 key 不存在时自动创建默认值
d["x"] += 1  # d["x"]只要出现，就直接增加这个键，自动创建键值为0
d = defaultdict(list)     # 默认值是 []
d["y"].append(42)  
```



---


# all
`all(iterable) 是 Python 内置函数`
传入一个“可迭代对象”（列表、生成器、集合、字符串、range 等）,逐个取元素，只要有一个元素为 False，就返回 False

`all([True, False, True])  # False`



---


# if 和 elif

连续写多个 if， 每个 if 都会各自独立判断

```
x = 5
if x > 0:
    print("正数")
if x > 3:
    print("大于 3")
if x < 10:
    print("小于 10")
```


`if / elif / elif` 是互斥分支，某个符合后面的不再执行，换句话说只可能执行其中的一个条件
```
x = 5
if x > 0:
    print("正数")
elif x > 3:
    print("大于 3")
elif x < 10:
    print("小于 10")
```




---


# try

#### try 和 exception
try 最少要搭配一个 except 或者 finally，比如下面两个例子
```
1.
try:
    pass
finally:
    pass

2.
try:
    pass
except Exception:
    pass
```

try 表示先对 try 中的进行尝试，如果 try 中有问题，会往多个 except 中逐个匹配错误条件，匹配到哪个，就去执行这个 except 中的代码；就理解成：执行....除了....

```
try:
    print("A")         # A 会被打印
    x = 1 / 0          # 这里抛出 ZeroDivisionError
    print("B")         # B 不会被打印
except ValueError:
    pass
这一段报错，因为出现除零错误，但是进行错误排除没找到对应的错误项，最终整段报错


try:
    print("A")
    x = 1 / 0          
    print("B")
except ValueError:
    pass
except ZeroDivisionError:
    pass
上面这段修改后的就不会报错了

try:
    print("A")
    x = 1 / 0          
    print("B")
except Exception:
    pass
上面这段也是可以的，Exception 涵盖大多数错误


try:
    print("A")
    x = 1 / 0         
    print("B")
except Exception as e:
    print(type(e))    # <class 'ZeroDivisionError'>


```

#### else
只有 try 中完全没问题，else 中的东西才会被执行，
```
try:
    print("A")
    x = 1 / 0         
    print("B")
except Exception as e:
    print(type(e))
else:
    print("else")    # 不被执行
    

```

#### finally
严格顺序要求如下：
```
try:
    ...
except ...:
    ...
except ...:
    ...
else:
    ...
finally:
    ...
```

`finally` 中的内容，不论 try 对不对，不论有没有 exception 匹配上，都会被执行
```
try:
    print("A")         # A 会被打印
    x = 1 / 0          # 这里抛出 ZeroDivisionError
    print("B")         
except ValueError:
    pass
finally:
    print("finally")
这一段会报错，因为 try 中不对，同时错误也没被 except 匹配到，但是仍然正常打印了 "A" 和 "finally"
```


---


# `__slots__ = ('son', 'end')` 

不写 slots，实例化之后还可以单独加属性
```
class Node:
    def __init__(self):
        self.son = {}
        self.end = False

a = Node()
b = Node()

a.x = 1
a.__dict__ 结果为 {'son': {}, 'end': False, 'x': 1}
b.__dict__ 不受影响
```


slots 意为 槽位，写了之后就限制这个类只能有这两个属性，出现其他的就会报错

例1：
```python
class Node:
    __slots__ = ('son', 'end')

    def __init__(self):
        self.son = {}
        self.edn = False

a = Node()  # 实例化的时候就会自动执行 __init__, 报错 'Node' object has no attribute 'edn'
```

例2：
```python
class Node:
    __slots__ = ('son', 'end')

    def __init__(self):
        self.son = {}
        self.end = False

    def mark_end(self):
        self.x = True   

a = Node()  # 正常执行
a.mark_end()  # 报错
```

例3：
```python
class Node:
    __slots__ = ('son', 'end')

    def __init__(self):
        self.son = {}
        self.end = False  

a = Node()
a.x = 1  # 报错
```

---

# nonlocal，global 和 类属性

#### nonlocal

Python 中，函数里遇到一个名字，会按 **LEGB** 的顺序查找
```
L：Local（本函数局部）
E：Enclosing（外层函数的局部）
G：Global
B：Builtins（内置）
```
**nonlocal 限制只会去 外层函数的局部 E 去找，不会去找 G 和 B**，因此只能用于该函数外层还有一个函数的情况

只要一个名字在函数体内出现了赋值（包括 `=`、`+=` 等），其默认为该函数的 local

因此下面这段代码中，由于不存在赋值，按照 LEGB 的顺序，能够访问到函数外的 ans

```python
class Solution: 
 def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
     ans = []
     def dfs(root: Optional[TreeNode]):
         if not root:
             return
         dfs(root.left)
         dfs(root.right)
         ans.append(root.val)

     dfs(root)
     return ans
```


而下面这段代码由于出现了赋值 `ans = max(ans, left + right)`, 因此要声明 `nonlocal ans`, 否则就会被当成函数内部临时创建的 local 变量
```python
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:

        def dfs(root: Optional[TreeNode]):
            nonlocal ans
            if not root:
                return 0
            left = dfs(root.left)
            right = dfs(root.right)
            ans = max(ans, left + right)
            return max(left, right) + 1
        # 不要求外层中 ans 一定在 内层函数的前面，不过一般还是习惯写在前面
        ans = 0
        dfs(root)
        return ans
```


下面这种 grid 作为外层函数传入的参数，也是可以在内层中首次用到 grid 之前写上 nonlocal;
不过下面这种函数内部只是改变 grid 其中元素的值，grid 本身的 name binding 没有变，下面这个不写 nonlocal 
```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])

        def dfs(i: int, j: int) -> None:
            if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != "1":
                return
            grid[i][j] = "2"
            dfs(i, j - 1)
            dfs(i, j + 1)
            dfs(i - 1, j)
            dfs(i + 1, j)

        ans = 0
        for i, row in enumerate(grid):
            for k, e in enumerate(row):
                if grid[i][k] == "1":
                    ans += 1
                    dfs(i, k)
        return ans
```

#### global 用法

```
x = 0

def f():
    global x
    x += 1
    return x

f()
```

这里如果写 nonlocal x 就会报错  `no binding for nonlocal 'x' found`

```
x = 10

class A:
    def f(self):
        global x
        x += 1

a = A()
a.f()
```
这里不写 global 报错 `local variable 'x' referenced before assignment`


#### 类中定义的方法涉及的外层变量情况

```python
class Solution:
    head = None
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if root is None:
            return
        self.flatten(root.right)
        self.flatten(root.left)
        root.right = self.head
        root.left = None
        self.head = root
```

这里 head 在类中作为一个属性，self 是 类的蓝图代指，因此用的时候 `self.head` 即可拿到




---

# Python 换行便于阅读


Python 中：
**只要代码还在 `() / [] / {}` 里面，就可以自然换行，不需要写 `\`**

```python
from math import inf

# 函数参数的注解
class Solution:
    def isValidBST(
        self,
        root: TreeNode | None,
        left: float = -inf,
        right: float = inf,
    ) -> bool:
        if root is None:
            return True

        x = root.val
        # 逻辑表达式打包写在括号里面缩进换行就不用显式写 \ 了
        return (
            left < x < right
            and self.isValidBST(root.left, left, x)
            and self.isValidBST(root.right, x, right)
        )
```

```python
MAPPING = [
    "", "", "abc", "def", "ghi",
    "jkl", "mno", "pqrs", "tuv", "wxyz"
]
```




---


# 读写文件

```
with open("./part1.json", "w", encoding="utf-8") as file:
    file.write()
    
用 with 就不用手动关闭了，with 下面的缩进部分执行完会自动关闭文件
"r" 表示只读模式，如果第一个路径参数找不到文件就会报错
"w" 表示写入模式，如果第一个路径参数找不到文件就会新建一个；但是如果目录不存在就会报错；比如 with open("./files/part1.json", "w", encoding="utf-8") as file: 这句中，如果当前文件父目录下面没有 files 文件夹就会报错
```

```

```
---

# json.dump

**磁盘里的 JSON 文件：只是一串字符串**
json 本质上只是一种文本的组织形式，通过换行符号和空格，外层的{}也是字符串，加上原有的内容，从而可以实现字符串和 python 变量之间的转换；

json 支持的六种值类型（这里的类型指的不是 python 变量类型，但是六种 json 值类型相应可以转化成对应的 python 值类型）
- **object**：`{ "a": 1, "b": 2 }`  json 中没有 dict，但是当成 dict 理解，json写成的格式这个“键”必须是字符串形式的东西，虽然 json 文件本身也是一个字符串
- **array**：`[1, 2, 3]
以上两种 object 和 array 是允许内嵌的类型，因此常见的 json 文件最外层都是 {} 或 \[\]
- **string**：`"hello"`
- **number**：`123`、`3.14`（不区分 int/float；不允许 NaN/Infinity）
- **boolean**：`true` / `false`
- **null**：`null`

记忆：dump 是“倾倒”，就是导出到 json 文件，dumps 是导出 json 字符串
总之 dump 和 load 必须传文件对象 `fp`（file pointer），写在 with open 代码块里面；dumps 和 loads 不需要文件


#### json.dumps()
Python 对象 → JSON 字符串（返回 `str`）
```
data = {"name": "Alan", "nums": [1, 2, 3], "ok": True, "none": None}  
s = json.dumps(data, ensure_ascii=False, indent=2)

s 结果为 '{\n  "name": "Alan",\n  "nums": [\n    1,\n    2,\n    3\n  ],\n  "ok": true,\n  "none": null\n}' 
type(s) 为 str

indent 参数控制缩进，表示每次需要缩进的部分的空格数，比如下面的 indent = 2
{
··"name": "Alan",
··"nums": [
····1,
····2,
····3
··],
··"ok": true,
··"none": null
}
```

#### json.loads
JSON 字符串 → Python 对象（返回 dict/list/…）
```
data = {"name": "Alan", "nums": [1, 2, 3], "ok": True, "none": None}  
s = json.dumps(data, ensure_ascii=False, indent=2)
k = json.loads(s)    # k 又被转换回 dict 
```

#### json.dump() 
```
s = {"name": "Alan", "nums": [1, 2, 3], "ok": True, "none": None}
with open("./file.json", "w", encoding="utf-8") as file:
    json.dump(s, file, ensure_ascii=False, indent=4)
这里 open 之后 as，这样拿到一个文件对象，传入 json.dump， 另一个参数 s 即为 dict，总的逻辑就是把 python 中的 dict 数据类型写入一个 json 文件
w 写入模式为覆写，如果目标文件不存在则创建一个，但是无法创建文件夹；
```

#### json.load()
```
s = {"name": "Alan", "nums": [1, 2, 3], "ok": True, "none": None}  
with open("./file.json", "w", encoding="utf-8") as file:
     json.dump(s, file, ensure_ascii=False, indent=4)  

上面已经写好一个 json 文件之后就可以读取，然后拿到 python 变量
with open("./file.json", "r", encoding="utf-8") as file:
     k = json.load(file)
```



---



# 链表
```

from __future__ import annotations

class ListNode:
    def __init__(self, val: int = 0, next: ListNode | None = None) -> None:
        self.val = val
        self.next = next
        

# 1) 先创建三个“节点对象”（此时它们还没连起来）
n1 = ListNode(1)
n2 = ListNode(2)
n3 = ListNode(3)

# 2) 手动把 next 串起来：n1 -> n2 -> n3
n1.next = n2
n2.next = n3
n3.next = None   # 可写可不写，默认就是 None

head = n1  # 链表头指针/引用
# Python 中空链表表示为 head = None，相当于 C 里的指针 p = NULL

# 3) 遍历打印
p = head
while p is not None:
    print(p.val)
    p = p.next
```



---


# 二叉树

```
from __future__ import annotations

class TreeNode:
    def __init__(
        self,
        val: int = 0,
        left: TreeNode | None = None,
        right: TreeNode | None = None
    ) -> None:
        self.val = val
        self.left = left
        self.right = right
```





---




# continue, break, return，pass
`continue`只在 `for` 或 `while` 循环里有意义；
一旦执行到 `continue`，这一轮循环里 `continue` 后面的代码都会被跳过，不再执行，直接进入下一轮循环。

`break`立即结束离它最近的那一层 `for` 或 `while` 循环，外层不受影响
```
for i in range(n):
    lp, rp = i + 1, n - 1
    while lp < rp:
        ...
        else:
            s.add((nums[i], nums[lp], nums[rp]))
            break  # 退出 while 循环，外层的 for 循环不受影响
```

`return` 立即结束当前函数

```
普通函数里如果没有return (或只有return不跟表达式)，隐式 return None
def f():  
    pass  # 返回None  
def g():  
    return  # 返回None

Python 没有真正的多返回值语法糖，本质是返回一个 tuple
return a, b
# 等价于
return (a, b)
```

`pass` 放在任意缩进代码块都合法
```
if condition:
    do_something()
else:
    pass

try:
    pass
except Exception:
    pass
finally:
    pass
```


---



# Python 变量生命周期
Python 里只有 `函数/类/模块` 会创建新作用域（scope）
    出了`函数/类/模块` 作用域之后，其内部的变量的名字不能直接访问
`for/while/if` 这些“代码块”不会。变量是否“活着”，本质看有没有引用。
```
for i in range(5):
    continue
print(i)  # 4
```

# 原地操作
Python 中 In-place 不是看“变量名有没有变”，而是看“同一个对象的内容变没

id 不变 → 原地；id 变 → 新对象

不可变对象的 `.method` 一定是非原地操作，返回新的东西
可变类型`list`, `dict`, `set`，带`update / add / remove / clear / sort / reverse`的 `.method` 大概率都是原地操作

大多数 **list 自带的方法** 为原地
- `append`, `extend`, `insert`
- `pop`, `remove`, `clear`
- `sort`, `reverse`
```
a = [3,1,2]
a.sort()      # 原地
a.reverse()   # 原地
```
`pytorch` 中函数名后面有`_` 的，一般是原地操作 （ pytorch 自己的方法命名风格）
```
x.relu_()        # 原地 ReLU
x.zero_()        # 原地清零
x.copy_(y)       # 原地拷贝
param.grad.zero_()
```
绝大多数高层API调用都是非原地的，API 基本都是`y = f(x)`的风格，非原地输入不改，返回新对象是非常合理的


# 变量引用

Python中只有三种东西： 
- 对象（object）
    对象是真实存储，具有三元属性：id, type, value
- 名字（name）
    名字**不存储数据**，只承担“指向对象”的索引作用。
- 引用（reference）
     名字引用对象
```
a = [1, 2, 3]  # 让名字 a 引用这个list

c = [1, 2, 3] 
c = d  # 赋值只是让名字 d 也引用名字 c 所引用的东西, id(a) == id(b)
```


---


# 多重赋值
```
a, b = b, a  # 先把右边的值求完，然后打包成一个临时的tuple，解包给左边
# python中lp = 0, rp = len(height) - 1是错误写法，一般只有链式赋值和多变量赋值
# 链式赋值 lp = rp = 0
# lp, rp = 0, len(height) - 1  # 多重赋值
```

多重赋值的规则是： 右侧会先整体求值并打包成元组，左侧从左到右依次分配
```
nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
和
nums[i], nums[nums[i] - 1] = nums[nums[i] - 1], nums[i]
不等价；
nums[nums[i] - 1] 依赖于 nums[i]，所以 nums[i] 不能先改变，第一种写法是正确的
```


---



# .sort() 和 sorted()

大多数通用编程语言，如果你不指定排序方向，默认都是升序
`list.sort()` —— 列表的原地排序方法, 返回值是 `None`，只能用在 **list 对象** 上，默认升序
`sorted()` —— 通用的非原地排序函数，可以用在**任何可迭代对象**上，返回值为list, 默认升序
```
nums = [3, 1, 4, 2]
b = sorted(nums)  # 返回一个新列表,nums 还是 [3, 1, 4, 2]

b = sorted(nums, reverse=True)  # 降序
b = sorted(nums, key=abs)  # 按绝对值排序
d = {"b": 2, "a": 1}
keys_sorted = sorted(d)  # ['a', 'b'] 字典默认迭代键
```
`.sort()` 的参数 `key` 控制排序规则，如果不传入就按照默认规则
```
intervals = [[3, 5], [1, 2], [2, 4]]
intervals.sort(key=lambda x: x[0])
# 按照每个区间左端点升序排列，如果多个区间左端点相等，相对顺序会保持原来的输入顺序
# key=函数名，表示按"key(元素)"的结果默认升序排序
# 匿名函数本质就是创建一个函数对象，但不通过"def"命名，即只关注一次简单的操作，不关注函数名
```


---


# .reverse() 和 reversed()

`.reverse()`
```
a = [1, 2, 3]
a.reverse()  # 原地操作，时间O(n)，空间O(1)
type(a.reverse())  # 类型为 NoneType, 说明返回值是 None
```

`reversed()`
```
a = [1, 2, 3]
b = reversed(a)
type(b)  # list_reverseiterator, 说明返回的是一个迭代器
```


---


# range
```
range(start, stop, step)  # step=-1表示每次减1
for i in range(5)  # 0,1,2,3,4 从零开始，左闭右开
for i in range(1,5)  # 从1开始到5，左闭右开
for i in range(4, -3, -1):  # 从4开始，走到-3，左闭右开，每次减1；结果为[4, 3, 2, 1, 0, -1, -2]，同样是左闭右开
```




---

# enumerate

`enumerate(iterable, start=0)` 索引从 start 的传入值开始
迭代时同时拿到 "编号" 和 "元素"

```python
list = ['a', 'b', 'c']
for i, x in enumerate(list):
    print(i, x)
结果为 
0 a
1 b
2 c

list = ['a', 'b', 'c']
for i, x in enumerate(list，start = 9):
    print(i, x)
结果为
9 a
10 b
11 c
```


---



# pandas .csv 文件读取相关语法

.csv(Comma-Separated Values)，纯文本文件，每一行是一个样本，行内用comma分隔

**pd 怎么知道 csv 里的是什么数据类型？**
**CSV 文件** = 純文本，每个单元格就是字符串，比如 `"20"`、`"85.5"`、`"Alice"`。
`pd.read_csv` 读取文件，把文本按行、按列拆开; 对每一列进行**类型推断**。
类型推断就是一套固定的尝试顺序，bool → int → float → datetime → 其他 → object，如果尝试到哪个，这一列的所有非缺失值都匹配上，那就识别成该数据类型
当用 `df["col"]`、`df.loc`、`df.iloc` 去取数据时，取出来的对象就已经带有这个 dtype 了。
```
# .csv
name,age,score
Alice,20,85.5
Bob,21,90
Charlie,19,78

df = pd.read_csv(csv_path)  # df 数据类型为 pandas.core.frame.DataFrame
# 索引语法
df["name"]       # 取单列，返回 Series
df[["name","age"]]  # 取多列，返回 DataFrame
```

**Pandas.Series**
这是 pandas 自己定义的类；就是**带“索引”的一维数组**
**Series 的常见属性和基本用法**
```
s = df["Absorption_I"]
s[0]  # 跟list一样的索引方式
s.name       # 列名：'Absorption_I'
s.dtype      # 
s.shape      # (35102,)
s.index      # 行索引 RangeIndex(start=0, stop=35102, step=1)
s.to_numpy() # 推荐的方式：转成 numpy 数组
s.tolist()   # 转成 Python list
```

---



# `tensor.unsqueeze()`

n维张量，有n+1个间隔，unsqueeze方法可以传入参数范围为 $[0,n]$ 共 $n+1$ 个数，如三维`(N,H,W)`，可以理解为 间隔N间隔H间隔W间隔，那么`unsqueeze`可传入的参数为0,1,2,3
```
tensor.unsqueeze(0)  # 在N之前插入 → (1, N, H, W)
tensor.unsqueeze(2)  # 在H和W之间插入 → (N, H, 1, W)
```

---



# tensor 的存储和视图属性

**存储属性** 就是一个一维数组形式的连续的内存区域，底层存储形式
**视图属性(数学语义)** 控制按照什么顺序去读取那段连续的存储
```
storage_index = offset + i0*stride[0] + i1*stride[1] + ... + in*stride[n]
视图本质上就是：同一块 storage，配不同的 (size, stride, offset) 组合
stride() 返回的是一个元组，它告诉你：在访问某一个维度时，你需要在内存中跳跃多少个数据位置才能到达下一个元素。
```

**实际中，打印，数学运算，索引，`.shape` 或者 `.size()` 等操作，都是按照张量的 视图属性(数学语义) 进行的**
.size() 或 .shape 可以得到某个 tensor 的视图属性(数学语义)，
(B,H,W), tensor数学语义排列顺序, 最后两个维度永远都是高和宽, 批次维度 B 是(H,W)往后延伸
(N,B,H,W),在(B,H,W)的基础上往右延伸
**这种理解方式永远指的都是视图属性(数学语义)，与实际存储没有任何关系**

存储属性存在的意义是多个视图可以共享一块存储，节省存储

**contiguous.**  C语言和大多数深度学习框架，contiguous 的多维张量在实际内存中采用 row-major，从一个点开始，**先向右延伸，再向下延伸得到矩阵，再向后延伸得到三维张量**；数学含义和底层存储一致
```
torch.zeros(2, 3)
torch.ones(4, 5)
torch.rand(3, 4, 5)
torch.randn(10)
torch.arange(12).view(3, 4)
a = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 在内存中实际为一维的顺序 1, 2, 3, 4, 5, 6
这些方式创建的tensor都是contiguous的
可以用 a.contiguous() 判断
```

**view() reshape() transpose() permute() contiguous() 用法**
```
/.view() 强制要求张量 contiguous.view() 只改视图属性，不 copy
/.reshape() 如果张量contiguous，等价于view(), 如果不contiguous但是新shape和stride兼容，只改视图不copy; 不contiguous也不兼容，等价于.contiguous().view()

/.contiguous() 如果张量不contiguous，该方法按照视图逻辑 copy 一份 contiguous 的

/.permute() 和 .transpose() 二者都改变视图属性，不copy，不要求张量一定要是contiguous；
```
**使用上，只有用到`.view()`的时候，才用关注下是否 contiguous**
**我们实际写代码，根本不关注实际存储，始终关注的都是视图属性(数学语义/逻辑顺序)是什么样，某个运算作用在哪个维度**
- .permute() 和 .transpose() 无非是转置，shape和stride是兼容的，是一种换轴重排索引的操作而已，本质上就两种情况，两个维度转置，从不同视角观察三个维度所形成的立方体整体
- view() 和 reshape() 基本也可以理解为从 contiguous 出发，可以理解为把原本张量的逻辑顺序按照 row-major 的风格 (其实就是contiguous的风格，反正都可以理解为从 contiguous 出发) 拉直成1D，再按照 row-major 的风格变形成传入的参数形状
```
x = torch.tensor([
    [[ 0,  1,  2],
     [ 3,  4,  5]],
    [[ 6,  7,  8],
     [ 9, 10, 11]]
])  # 展平为0,1,2...11
x.reshape(-1,3)  
tensor([[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11]])  # 按照(4,3)自然排列出来了

```

**为什么permute这种看似随意变换维度顺序的操作是前后双射的?** 首先考虑2D，2D转置与否不改变数据本身内在结构，只是改变观察的视角；3D首先轮换对称不改变结构，如果轮换对称再加上二维的镜像反射，那么3个维度6种排列也是等价的，也相当于只改变观察的视角；4D开始可以把3D当成基本单元，重新开始嵌套，高维张量只是3D的嵌套，因此permute变换是合理的，能够正确映射过去映射回来的


# 高维 tensor 在实际项目中一定基于视图属性(数学含义)理解
最后两维度一定是高和宽，然后再往前读，(B,H,W)，就是(H,W)有了之后往后延伸，(N,B,H,W)是(B,H,W)往右延伸
```
torch.arange(24).view(4, 3, 2)
tensor([[[ 0,  1],
         [ 2,  3],
         [ 4,  5]],

        [[ 6,  7],
         [ 8,  9],
         [10, 11]],

        [[12, 13],
         [14, 15],
         [16, 17]],

        [[18, 19],
         [20, 21],
         [22, 23]]])
```

# Python annotation 
```
from __future__ import annotations  # 仅仅为了让类型注解更好用、兼容，与实际的代码逻辑无关

class A:
    def foo(self, other: B) -> B:  # 这里用到了 B
        return other
class B:
    ...
定义A的时候B还没定义，但是类型注解中提到了B，此时就要写from __future__ import annotations
```

```
x: int  # 效果等价于 x = 100
merges: dict[tuple[int, int], int] = {}  # 效果等价于merges= {}
def func(param: int, param2: str = "hi") -> bool:  
# def func(param, param2="hi"):

list[int] 长度不限，每个元素都必须是int。
tuple[int, str], 限定长度必须是2，分别是int和str
tuple[int, ...] 长度不限，每个元素必须是int

def foo(x: Tensor | None = None):  
# PyTorch 中很常见，表示这个参数你可以不传，不传默认为 None , 如果传，应该传 Tensor
# None 是 NoneType 的一种数值， NoneType 只有这一种数值。类似 9 是 int 的一种数值
```

# PyTorch 数据精度和 device 指定

## float32
**PyTorch 里训练模型，几乎所有 tensor 都用 `float32`**, 另外 `tensor` 中的元素类型必须是数值型的，单个 `tensor` 内部的 `dtype` 一定是一致的(比如都是`float32`)

比如 `randn(*size, *, generator=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) -> Tensor`
虽然 `dtype=None` 以及 `device=None` ，但若是没有特殊指定，就默认为`float32` 和 `cpu`
```
None 是 NoneType 的一种数值， NoneType 只有这一种数值。类似 9 是 int 的一种数值
表示这个参数你可以不传，不传默认为 None , 常常作为哨兵值用于内部的特定逻辑
def pad(input, pad, mode='constant', value=None):
    if mode == 'constant':
        if value is None:
            value = 0.0
        return _constant_pad_impl(input, pad, float(value))
    else:
        return _other_mode_pad_impl(input, pad, mode)
```

只有一些传统的数值分析比如反复求逆、特征值问题等，或者有迹象表明是数值不稳定的问题时，才会考虑去设置为更高精度的`float64`

---



## int64
在 PyTorch 里，几乎所有 “离散 id / 标签 / 下标” 都用
PyTorch 统一规定张量索引类型就是 int64。
```
# 两种写法都行
dtype=torch.int64 
dtype=torch.long
torch.long is torch.int64   # True
# A is B 为 True，表示 A 和 B 指向同一个对象
```
但是Python 3 中只有`int`，表示任意精度整数，`torch.float32` ,`torch.int64` 这些都是 torch 库内部自己定义的

---



## .to(device) 把 tensor 复制到目标设备(返回新张量)
```
# 全局使用判断逻辑给出的 device 变量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 把 model(nn.Module) 中登记的所有 nn.Parameter 和 buffer 转到 cuda 上
model = model.to(device)  
# 特征和标签转到 cuda 上
X_batch = X_batch.to(device)      
y_batch = y_batch.to(device)
# 自己创建的其他的张量转到 cuda 上
a = torch.zeros(batch_size, hidden_size)
a = a.to(device)  
```

---



# Padding

序列任务(输入输出是有顺序的一串东西)在尾端补
```
import torch.nn.functional as F
F.pad(input, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))
# input 为传入的张量，后面的 tuple 中每个数字表示在相应位置填充的长度, 是按照 (B,H,W) 从后往前数的，也就是高维 tensor 数学含义的空间理解方式，先往右，再往下，再往后...
# 默认 mode='constant'时, value=0, 填充位置自动转换为和 input.dtype 一样的类型

x = torch.ones((1, 2, 3), dtype=torch.float32)
y = F.pad(x, (0,0,0,0,1,2))
# tensor([[[1., 1., 1.],
         [1., 1., 1.]]])
# tensor([[[0., 0., 0.],
         [0., 0., 0.]],

        [[1., 1., 1.],
         [1., 1., 1.]],

        [[0., 0., 0.],
         [0., 0., 0.]],

        [[0., 0., 0.],
         [0., 0., 0.]]])
```

**1D 序列 shapelet 切分为多个 Patch 的写法**
具体代码见`Shapelet`项目中的`sdd.py`文件
固定 patch 长度
```
# 传入参数 patch_len
num_patches = math.ceil(shapelet_length / patch_len)  # 向上取整
total_len = num_patches * patch_len
pad_len = total_len - L
shapelet_padded = F.pad(shapelets_raw, (0, pad_len))
# def forward(self, shapelets_raw: torch.Tensor) -> torch.Tensor: 中
patches = shapelet_padded.view(N, self.num_patches, self.patch_len)
```
固定 patch 数量
```
# 传入参数 num_patches
patch_len = math.ceil(shapelet_length / num_patches)
total_len = num_patches * patch_len
pad_len = total_len - L
shapelet_padded = F.pad(shapelets_raw, (0, pad_len))
# def forward(self, shapelets_raw: torch.Tensor) -> torch.Tensor: 中
patches = shapelet_padded.view(N, self.num_patches, self.patch_len)
```
最后编码完成的数据要去掉 padding 部分重新索引


# epoch
```
for epoch in range(E):
    train_one_epoch()  #  训练集上更新参数
    val_metrics = validate()  # 验证集上只前向传播计算 metrics
    if val_metrics better than best:
        save_checkpoint(best)
load_checkpoint(best)
test(best)  # 测试集在开发过程中永远是不会使用的
```

# run.py
```
if __name__ == "__main__":
    main()
# run.py 作为脚本直接执行的时候，其 __name__ 为"__main__"; 作为模块被别的文件 import时, 比如出现 import run, 其 __name__ 为"run"(模块名)
# 如果不写这个判断语句，import 这个文件的时候也会把这个文件中所有的顶层代码执行一遍
# 顶层代码: 不在任何函数/类内部
```

# Dataset 和 Dataloader
```
from torch.utils.data import Dataset, DataLoader

# 先自定义数据集，以下三个方法必须写
class MyData(Dataset):  
    def __init__(self, root_dir, label_dir):  
        self.root_dir = root_dir  
        self.label_dir = label_dir  
        self.path = os.path.join(self.root_dir, self.label_dir)  
        self.img_path = os.listdir(self.path)  
    
    def __getitem__(self, idx):  
        img_name = self.img_path[idx]  
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)  
        img = Image.open(img_item_path)  
        label = self.label_dir  
        return img, label  
    
    def __len__(self):  
        return len(self.img_path)
# __getitem__定义每一条数据具体是什么，让数据集 subscriptable/indexable，实例化后，MyData[i]语法会自动调用.__getitem__(i)

# 实例化自己定义的数据集
dataset = MyData(
    root_dir="data/train",   
    label_dir="cat",        
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,                  
    num_workers=4,  # 多进程
    pin_memory=torch.cuda.is_available(),  # cuda 上这个开了就有收益，cpu约等于没有，记住就行，不用管原理
    drop_last=True,  # 丢掉最后不足 batch 的部分
)
# Dataloader 返回一个可迭代对象，是一个发牌官，他自身无法 subscriptable, 他会调用传入 dataset 的 len 方法，然后得到索引，把索引传入 dataset 的 __getitem__ 中，就能得到数据集中单条数据；单个数据为基本单位，shuffle之后拼成 batch


for imgs, labels in loader:
    imgs = imgs.to(device, non_blocking=True)
``` 
dataset 定义的 `__getitem__` 返回什么结构，for迭代dataloader就拿到什么结构，多一个批次维度而已
```
class MetaDataset(Dataset):
    """
    读取 CSV 中的 Absorption_I / Absorption_M, 转换为 (1,100) 张量。
    """
    
    def __init__(self, csv_path: str) -> None:
        super().__init__()
        df = pd.read_csv(csv_path)
        self.abs_I = df["Absorption_I"].apply(self._parse_vec).tolist() 
        self.abs_M = df["Absorption_M"].apply(self._parse_vec).tolist()  

    @staticmethod
    def _parse_vec(raw: str) -> torch.Tensor:
        # 字符串形如 "[0.78 0.89 ...]"，用 numpy.fromstring 解析为长度 100 向量。
        arr = np.fromstring(raw.strip("[]"), sep=" ")
        if arr.shape[0] != 100:
            raise ValueError(f"光谱长度不是 100，实际 {arr.shape[0]}")
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1,100)

    def __len__(self) -> int:
        return len(self.abs_I)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.abs_I[idx], self.abs_M[idx]
        
train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=torch.cuda.is_available(),
)

for xI, xM in train_loader:
            xI = xI.to(device)
            xM = xM.to(device)
```
这个例子中__getitem__不返回索引，返回俩张量，那loader迭代的时候，得到的值也是俩张量
`__getitem__`返回啥，for 迭代 dataloader的时候就相应得到啥


# callable

像函数一样用圆括号调用，定义了__call__方法就能callable
```
class Scaler:
    def __init__(self, k: float):
        self.k = k
        
    def __call__(self, x: float) -> float:
        return self.k * x
s = Scaler(3.0)
s(2.0)        # 6.0
callable(s)   # True
```
`nn.Module` 子类中 `forward` 与 callable 的关系：
`nn.Module` 已经实现了 `__call__`，它会做封装，最后调用 `forward`, 因此自己写的模型中的模块中的 `forward` 的逻辑就是 `callable` 的结果
```
proj = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = proj(input)  # callable
```


# iterable

某个对象 iterable 等价于 能写成`for x in obj`
iterable 的对象内部需要实现 `__iter__` 或 `__getitem__` 方法

`DataLoader` 之所以用下面的`for` 语法
`for xI, xM in train_loader:
            xI = xI.to(device)
            xM = xM.to(device)`
是因为他类定义中实现了`__iter__`
可以认为
`__iter__` 中通过 `len(dataset)` 得到 `indices`，再调用 `dataset` 的 `__getitem__` 取出对应样本，组成`batch`，因此对`dataloader`每次迭代拿到的东西看的是传入`dataset`中定义的`__getitem__` 的返回值



# 类定义中的 self
**`self`就是一个未来实例的“代称”，实例化之后，`self`就是实例对象本身**
`self` 服务于实例属性(区别于类属性)和实例方法(区别于`@classmethod` 和 `@staticmethod`)
```
class SimpleNet(nn.Module):
    class_var = "类属性"  # 类属性  
    def __init__(self):  # 实例属性  
        super().__init__()  
        self.fc1 = nn.Linear(10, 5)  
    def print_self_info(self):  
        print(f"self的类型: {type(self)}")  
        print(f"self的id: {id(self)}")  
        print(f"self就是实例本身: {self is model}")  

model = SimpleNet()  
print(f"model的id: {id(model)}")  
model.print_self_info()  # True
```
能够验证：实例化之后，`self` 就是实例对象本身；

并且，类的定义的过程是一个蓝图，在这个过程中，既然`self`是未来实例对象的代称，那么在蓝图里面`self.method()`自然是可行的，因为反正是蓝图，真调用说明已经实例化了，那都已经实例化了`self`就是实例化对象，当然能够调用相应方法


# `@staticmethod` 静态方法

静态方法逻辑不依赖 `self` 的任何状态，但是往往又是只有某一个类会用到，就写在这个类里面了

静态方法调用不强制要求实例化，实不实例化都行
```
class MetaDataset(Dataset):

    def __init__(self, csv_path: str) -> None:
        super().__init__()
        df = pd.read_csv(csv_path)
        self.abs_I = df["Absorption_I"].apply(self._parse_vec).tolist() 
        self.abs_M = df["Absorption_M"].apply(self._parse_vec).tolist()  

    @staticmethod
    def _parse_vec(raw: str) -> torch.Tensor:
    # 前缀下划线表示内部工具，用于实现细节；外部代码最好不要依赖它
        arr = np.fromstring(raw.strip("[]"), sep=" ")
        if arr.shape[0] != 100:
            raise ValueError(f"光谱长度不是 100，实际 {arr.shape[0]}")
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # (1,100)

    def __len__(self) -> int:
        return len(self.abs_I)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.abs_I[idx], self.abs_M[idx]
```
比如这里`self.abs_I = df["Absorption_I"].apply(self._parse_vec).tolist()` 中的`self._parse_vec` 就是实例化的写法，`apply(MetaDataset._parse_vec)` 这种写法也是可以的，即直接用类名调用就行


# loss 和 optimzier  未完待续

#### 计算图  

**同时满足以下两点，PyTorch 会记录计算图：
1. **发生了张量运算**（加法、matmul、conv、索引、norm……任何产生新 Tensor 的算子）
    
2. 这些运算在执行时，若grad 追踪开启(`torch.is_grad_enabled() == True`), 并且该运算的输入里至少有一个 Tensor 需要梯度(`requires_grad=True`）

---

`torch.is_grad_enabled()` 是 `PyTorch autograd` 机制维护的一个运行时开关，和 Python 语言本身无关。只有在用 torch 的 Tensor 运算时，这个开关才有意义；
在一个正常的 Python 进程里，PyTorch 的 grad mode 默认就是True。 只有在显式写了这些东西时，才会把它关掉：  
```
with torch.no_grad():   # with 代码块内 False
with torch.inference_mode():  # with 代码块内 False
torch.set_grad_enabled(False)  # 当前线程直接设置为 False 并保持，直到再设置回 True
```
---

`.backward()` 不是去延迟创建图，而是沿着已经记录好的计算图把梯度写进符合要求的 tensor 的 `.grad` 中
一次`.backward()` 结束后计算图就会被释放

任何 tensor 都有 `.grad` 属性，`.grad` 也是一个张量，默认为 `None`；某个 tensor **只有满足 `requires_grad=True` 且在计算图中为叶子节点**，其 `.grad` 才会在 loss 回传的时候写入相应数值；中间变量会计算梯度用于链式法则反向传播，但是不会写入`.grad`

---



#### 规范书写 class model(nn.Module) 中的参数和常量

- **Python 常量** (int/float/bool/str、超参、开关）
    在_`_init__` 中写 self.constant = ...
    
- **Tensor 参数** 
    在`__init__` 中写 `self.xxx = nn.Parameter(...)` 或放在子模块里(`nn.Linear` 等），这些张量内部写好了 `requires_grad=True`，所以最终写成`optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)` 传入优化器，就能很方便地更新参数
    
- **非参数张量** (在 forward中会用到的 Tensor，但不希望它作为参数被优化器更新）
    在`__init__` 中写`self.register_buffer(...)`

---
```
self.register_buffer("std", torch.ones(d), persistent = True)
效果上等同于 self.std = torch.ones(d) 的同时，把非参数张量登记为 Module 的 buffer，之后会随 model.to(device/dtype) 等自动迁移(跟参数类似)，persistent 默认为 True，表示让这个 buffer 进入 model.state_dict()
```
---

常见的非参数张量：
- 模型状态类(会变，但不靠梯度学): BatchNorm 的 running_mean 等
- 常量张量: 不训练、但要跟随 `.to()` 走设备/精度；往往希望随 `model.state_dict()` 保存, 比如，输入归一化用的 mean/std, 固定位置编码
- rebuildable 中间结构：不需要从 checkpoint 里恢复，也不需要梯度，在运行时可以用一套确定的规则（通常只依赖配置 + 输入的 shape/device/dtype）把它再生成出来，生成成本可接受，比如因果注意力中的上三角矩阵 `causal mask`

---



#### loss 反向传播

0 维 tensor 才有`.backward()`方法
```
# 0维张量没有维度，表示一个单一的数值, 
x= torch.tensor(3.0)
x.shape  # torch.Size([])

# 1维张量类似数组
x= torch.tensor([3.0])
x.shape  # torch.Size([1])
```
---

每个 batch 调用 `loss.backward()` 之前，要调用一次 `optim.zero_grad()` , 
原因是，`Pytorch` 中自动求导的梯度是累加的，正是这种累加实现了不同分支出去的梯度能够按照数学规则正确相加，**但是这要求我们每个批次计算的时候手动清零一次**

---



#### optimizer 如何拿到梯度

`loss.backward()`：沿图反传，在每个叶子参数 tensor 处累积梯度，写入`.grad`
`optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)` 就可以拿到传入参数的`.grad`，然后 `optimizer.step()` 用拿到的梯度进行参数更新

---


#### loss, optimizer, model 通用训练框架  
```
criterion = nn.XXLoss()
model = SDDModel(...).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

for _ in range(1, args.epochs + 1):
    model.train()
    for x, labels in train_loader:
        x = x.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        output = model(x)
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()
```

---



# with 语法
进入时做一次事，退出时再做一次事，一定会执行退出逻辑，即使 with 内部抛异常
```
# 推理阶段不构建计算图，省内存

with torch.no_grad():
    y = model(x)
    
# torch.no_grad() 只在这个 with 代码块内部生效；一旦退出这个块，恢复到进入 with 之前的值
```


# argparse  未完待续
















# `torch.nn.functional`


`F.cross_entropy`（函数式）`nn.CrossEntropyLoss`（模块式）

`nn.functional` 往往只提供算子
- 不持有可训练参数
- 也不保存超参数配置（每次调用都把配置和数据一起传入）
- 适合“就地计算”“快速实验”“动态配置”

`nn.CrossEntropyLoss` 模块式
- 可以持有可训练参数
- 也可以持有固定的超参数配置（如 `nn.Dropout` 的 `p`）
- 实例化时完成配置，forward 时只传入数据，代码更整洁、更便于工程化


---



# `.json` `.jsonl`


`.json`: 一个 list 里装多个 JSON object,  { ... } 里面包一些 `key-value pairs` 就是
一个 JSON object ( key 必须是字符串格式，value 可以是多种类型)

`.json` 文件示例，其中 "description" 的 value 是一个空字符串

```json
[
  {
    "sample_index": 1,
    "spectrum_tokens": ["<spec_0046>", "<spec_0184>", "<spec_0035>"],
    "description": "",
    "structure_sentence": "The material structure from top to bottom is TiO2_4900 nm, ZnS_3680 nm, Ta2O5_1000 nm, VO2_760 nm, ZnS_2800 nm, Nb2O5_930 nm, ZnS_2400 nm."
  },
  {
    "sample_index": 2,
    "spectrum_tokens": ["<spec_0161>", "<spec_0190>", "<spec_0072>"],
    "description": "",
    "structure_sentence": "The material structure from top to bottom is PDMS_290 nm, MgO_1560 nm, SiO2_4370 nm, VO2_2230 nm, ZnSe_660 nm, MgO_820 nm, BaF2_50 nm, MgO_940 nm, Al2O3_2270 nm, ZnS_30 nm, W_4920 nm."
  }
]
```


`.jsonl` 文件示例
```jsonl
{"sample_index": 1, "spectrum_tokens": ["<spec_0046>", "<spec_0184>", "<spec_0035>"], "description": "", "structure_sentence": "The material structure from top to bottom is TiO2_4900 nm, ZnS_3680 nm, Ta2O5_1000 nm, VO2_760 nm, ZnS_2800 nm, Nb2O5_930 nm, ZnS_2400 nm."}
{"sample_index": 2, "spectrum_tokens": ["<spec_0161>", "<spec_0190>", "<spec_0072>"], "description": "", "structure_sentence": "The material structure from top to bottom is PDMS_290 nm, MgO_1560 nm, SiO2_4370 nm, VO2_2230 nm, ZnSe_660 nm, MgO_820 nm, BaF2_50 nm, MgO_940 nm, Al2O3_2270 nm, ZnS_30 nm, W_4920 nm."}
```

文件里每一行都是完整且独立的 JSON object
`.json` 和 `.jsonl` 两者都用 JSON 语法来表达对象，单条样本的内容字段完全可以一样；只是读取工具不同，`.jsonl` 在 LLM 上更方便


---


