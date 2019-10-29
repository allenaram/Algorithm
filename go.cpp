#include <iostream>
#include <string>
#include <vector>
#include <list>
#include <stack>
#include <queue>
#include <algorithm>
#include <bitset>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <cmath>
#include <cstdlib>
#include <unordered_map>


using namespace std;

struct Node
{
	int val;
	Node* next;
	Node(int value, Node* node)
	{
		val = value;
		next = node;
	}
};
struct BTNode
{
	int val;
	BTNode* left;
	BTNode* right;
	BTNode(int value, BTNode* l, BTNode* r)
	{
		val = value;
		left = l;
		right = r;
	}
};



/*-------------------------------                     
        JianZhi 7       
--------------------------------*/
struct BTNode7
{
	int val;
	BTNode7* left;
	BTNode7* right;
};
class JianZhi7
{
public:
	BTNode7* Rebuild(vector<int>& pre, vector<int>& in)
	{
		return process(pre, in, 0, 0, pre.size());
	}
	BTNode7* process(vector<int>& pre, vector<int>& in, int preLeft, int inLeft, int length)
	{
		if (length == 0)
			return NULL;

		BTNode7* root = new BTNode7;
		for (int i = 0; i < length; i++)
		{
			if (pre[preLeft] == in[inLeft + i])
			{
				root->val = pre[preLeft];
				root->left = process(pre, in, preLeft + 1, inLeft, i);
				root->right = process(pre, in, preLeft + i + 1, inLeft + i + 1, length - i - 1);
				break;
			}
		}
		return root;
	}
};


/*-------------------------------
		JianZhi 8
--------------------------------*/
struct BTNode8
{
	int val;
	BTNode8* left;
	BTNode8* right;
	BTNode8* parent;
};
class JianZhi8
{
public:
	void findNext(BTNode8* cur)
	{
		BTNode8* nxt = NULL;
		if (cur->right)
		{
			cur = cur->right;
			while (cur->left)
			{
				cur = cur->left;
			}
			cout<<cur->val;
		}
		else if (cur->parent && cur->parent->left == cur)
		{
			cout << cur->parent->val;
		}
		else
		{
			while (cur->parent && cur->parent->right == cur)
			{
				cur = cur->parent;
			}
			cout << cur->parent->val;
		}
	}
};


/*-------------------------------
		JianZhi 10_2
--------------------------------*/
class JianZhi10_2
{
public:
	/*递归*/
	int process(int n, int restStep)
	{
		if (restStep < 0)
			return 0;
		else if (restStep == 0)
			return 1;

		return process(n, restStep - 1) + process(n, restStep - 2);
	}
	int cnt(int n)
	{
		return process(n, n);
	}
};


/*-------------------------------
		JianZhi 11
--------------------------------*/
class JianZhi11
{
public:
	int findMin(vector<int>& input)
	{
		int start = 0;
		int end = input.size() - 1;
		int mid = start;
		while (input[start] >= input[end])
		{
			if (end - start == 1)
			{
				mid = end;
				break;
			}
			mid = (start + end) >> 1;
			/*出现中间等与头等于尾的情况时，只能遍历*/
			if (input[start] == input[mid] && input[start] == input[end])
				return tranversing(input, start, end);
			/*缩小一半范围*/
			if (input[mid] >= input[start])
				start = mid;
			else if (input[end] >= input[mid])
				end = mid;
		}
		return input[mid];
	}
	int tranversing(vector<int>& input, int start, int end)
	{
		int minV = input[start];
		for (int i = start; i <= end; i++)
			minV = minV > input[i] ? input[i] : minV;
		return minV;
	}
};


/*-------------------------------
		快排
--------------------------------*/
class QuickSort
{
public:
	void mySwap(int* a, int* b)
	{
		int temp = *a;
		*a = *b;
		*b = temp;
	}
	vector<int> process(vector<int>& input, int start, int end)
	{
		int target = input[end];
		int less = start - 1; 
		int more = end + 1;
		int cur = start;
		while (cur < more)
		{
			if (input[cur] < target)
				mySwap(&input[cur++], &input[++less]);
			else if (input[cur] == target)
				cur++;
			else
				mySwap(&input[cur], &input[--more]);
		}
		vector<int> res = { less, more };
		return res;
	}
	void quickSort(vector<int>& input, int start, int end)
	{
		vector<int> res = process(input, start, end);
		int less = res[0];
		int more = res[1];
		if (less > start)
			quickSort(input, start, less);
		if (more < end)
			quickSort(input, more, end);
	}
};


/*-------------------------------
		JianZhi 12
--------------------------------*/
class JianZhi12
{
public:
	bool process(vector<vector<int>>& m, vector<vector<bool>>& mark, string& str, int cur, int i, int j)
	{
		if (mark[i][j])//走过的路不能再走
			return false;
		if (i < 0 || i >= m.size() || j < 0 || j >= m[0].size())//不能超出范围
			return false;
		if (str[cur] != m[i][j])//当前路径已不满足
			return false;
		else if (cur == str.size() - 1)
			return true;

		mark[i][j] = true;
		bool found = process(m, mark, str, cur + 1, i + 1, j) || 
			         process(m, mark, str, cur + 1, i - 1, j) ||
					 process(m, mark, str, cur + 1, i, j + 1) ||
					 process(m, mark, str, cur + 1, i, j - 1);
		if (!found)
			mark[i][j] = false;
		return found;
	}
	bool search(vector<vector<int>>& m, string str)
	{
		vector<vector<bool>> mark(m.size(), vector<bool>(m[0].size(), false));
		bool found = false;
		for (int i = 0; i<m.size(); i++)
			for(int j=0; j<m[0].size(); j++)
				if (process(m, mark, str, 0, i, j))
				{
					found = true;
					break;
				}
		return found;
	}
};


/*-------------------------------
		JianZhi 13
--------------------------------*/
class JianZhi13
{
public:

};


/*-------------------------------
		JianZhi 14
--------------------------------*/
class JianZhi14
{
public:
	/*递归*/
	int product(int rest)
	{
		if (rest == 0 || rest == 1)
			return 1;
		if (rest == 2)
			return 2;
		if (rest == 3)
			return 3;

		int curP = 1;
		int maxP = 1;
		for (int i = 1; i < rest; i++)
		{
			curP = i * product(rest - i);
			maxP = max(curP, maxP);
		}
			
		return maxP;
	}
	/*动态规划*/
	int product2(int rest)
	{
		vector<int> dp(rest+1, 1);
		for (int i = 1; i <= rest; i++)
			for (int j = 1; j <= i; j++)
				 dp[i] = max(j * dp[i-j], dp[i]);

		return dp[rest];
	}
};


/*-------------------------------
		JianZhi 16
--------------------------------*/
class JianZhi16
{
public:
	int process(double base, int exponent)
	{
		if (exponent == 0)
			return 0;
		if (exponent == 1)
			return base; 

		double result = process(base, exponent >> 1);
		result *= result;
		if (exponent & 1 == 1)
			result *= base;
		return result;
	}
};


/*-------------------------------
		JianZhi 17
--------------------------------*/
class JianZhi17
{
public:
	void PrintNum(int n)
	{
		vector<char> num(n, '0');
		while (increase(num))
		{
			print(num);
		}
	}
private:
	bool increase(vector<char>& num)
	{
		int carry = 0;
		for (int i = num.size() - 1; i >= 0; i--)
		{
			int sum = 0;
			if (i == num.size() - 1)
				sum = 1;
			sum += (num[i]-'0'+ carry);
			
			if (sum >= 10)
			{
				if (i == 0)
					return false;
				sum -= 10;
				carry = 1;
			}
			else
				carry = 0;
			num[i] = '0' + sum;
		}
		return true;
	}
	void print(vector<char> num)
	{
		bool startflag = false;
		for (int i = 0; i < num.size(); i++)
		{
			if (!startflag && num[i] != '0')
				startflag = true;
			if (startflag)
				cout << num[i];
		}
		cout << endl;
	}
};


/*-------------------------------
		JianZhi 18_2
--------------------------------*/
struct Node18_2
{
	int val;
	Node18_2* next;
	Node18_2(int value, Node18_2* node)
	{
		val = value;
		next = node;
	}
};
class JianZhi18_2
{
public:
	void DeleteNode(Node18_2** head)
	{
		Node18_2* root = new Node18_2(0, *head);//设置头结点
		Node18_2* pre = root;
		Node18_2* cur = *head;
		while (cur)
		{
			while (cur->next && pre->next->val == cur->next->val)
			{
				cur = cur->next;
			}
			if (pre->next != cur)
			{
				/*
				...//释放重复结点
				*/
				pre->next = cur->next;
			}
			else
			{
				pre = cur;
				cur = cur->next;
			}
		}
	}
};


/*-------------------------------
		JianZhi 19
--------------------------------*/
class JianZhi19
{
public:
	bool process(string& pattern, string& str, int curPattern, int curStr)
	{
		if (curPattern == pattern.size() && curStr <str.size())
			return false;

		if (curPattern == pattern.size() && curStr == str.size())
			return true;

		bool nextIsStar = curPattern + 1 < pattern.size() && pattern[curPattern + 1] == '*';
		if (curStr <str.size() && (pattern[curPattern] == str[curStr] || pattern[curPattern] == '.'))//当前字符匹配
		{
			if (nextIsStar)//当前模式为“字符*”
				return process(pattern, str, curPattern, curStr + 1) || 
					   process(pattern, str, curPattern + 2, curStr + 1) ||
					   process(pattern, str, curPattern + 2, curStr);
			else //普通匹配
				return process(pattern, str, curPattern + 1, curStr + 1);
		}
		else if (nextIsStar)
		{
			return process(pattern, str, curPattern + 2, curStr);
		}

		return false; //当前字符不匹配
	}
};


/*-------------------------------
		JianZhi 20
--------------------------------*/
class JianZhi20
{
public:
	bool checkNumber(string& str)
	{
		int cur = 0;
		bool integerPartFound = scanInt(str, &cur);
		bool decimalPartFound = false;
		if (cur <str.size() && str[cur] == '.')
			decimalPartFound = scanUnsignedInt(str, &(++cur));
		bool baseFound = integerPartFound || decimalPartFound;

		bool hasExponent = false;
		bool exponentFound = false;
		if (cur < str.size() && (str[cur] == 'e' || str[cur] == 'E'))
		{
			hasExponent = true;
			exponentFound = scanInt(str, &(++cur));
		}
		/*    匹配条件：1.没有无法匹配的字符串         */
		/*              2.必须有底数                   */
		/*              3.如果有'e'或'E'，则必须有指数 */
		if (cur == str.size() && baseFound && (!hasExponent || hasExponent && exponentFound))
			return true;

		return false;
	}
private:
	bool scanUnsignedInt(string& str, int* cur)
	{
		int pre = *cur;
		while ((*cur) < str.size() && (str[*cur] >= '0' && str[*cur] <= '9'))
			(*cur)++;

		return pre < *cur;
	}
	bool scanInt(string& str, int* cur)
	{
		if (str[*cur] == '+' || str[*cur] == '-')
			(*cur)++;
		return scanUnsignedInt(str, cur);
	}
};


/*-------------------------------
		JianZhi 21
--------------------------------*/
class JianZhi21
{
public:
	void Reorder(vector<int>& input)
	{
		int odd = -1;
		int cur = 0;
		while (cur < input.size())
		{
			if (input[cur] & 1 == 1)
				swap(input[cur], input[++odd]);
			cur++;
		}
	}
};


/*-------------------------------
		JianZhi 22
--------------------------------*/
struct Node22
{
	int val;
	Node22* next;
	Node22(int value, Node22* node)
	{
		val = value;
		next = node;
	}
};
class JianZhi22
{
public:
	Node22* GetNode(Node22* head, int n)
	{
		if (!head)
			throw new std::exception("当前链表为空");

		Node22* forward = head;
		Node22* back = head;
		for (int i = 0; i < n - 1; i++)
		{
			if (forward->next)
				forward = forward->next;
			else
				throw new std::exception("没有那么多结点！");
		}

		while (forward->next)
		{
			forward = forward->next;
			back = back->next;
		}
		return back;
	}
};


/*-------------------------------
		JianZhi 23
--------------------------------*/
struct Node23
{
	int val;
	Node23* next;
	Node23(int value, Node23* node)
	{
		val = value; 
		next = node;
	}
};
class JianZhi23
{
public:
	Node23* FindEntranceNode(Node23* head)
	{
		if (!head)
			return NULL;

		Node23* forward = head;
		Node23* back = head;
		/*找环上一点*/
		bool findCircle = false;
		while (forward->next && forward->next->next)
		{
			forward = forward->next->next;
			back = back->next;

			if (forward == back)
			{
				findCircle = true;
				break;
			}
		}
		if (!findCircle)
			return NULL;
		/*计算环上结点个数*/
		int cnt = 1;
		forward = forward->next;
		while (forward != back)
		{
			forward = forward->next;
			cnt++;
		}
		/*找环入口点*/
		forward = head;
		back = head;
		for (int i = 0; i < cnt; i++)
			forward = forward->next;
		while (forward != back)
		{
			forward = forward->next;
			back = back->next;
		}
		return back;
	}
};


/*-------------------------------
		JianZhi 24
--------------------------------*/
struct Node24
{
	int val;
	Node24* next;
	Node24(int value, Node24* node)
	{
		val = value;
		next = node;
	}
};
class JianZhi24
{
public:
	Node24* ReverseList(Node24* head)
	{
		if (!head)
			return NULL;

		Node24* pre = head;
		Node24* cur = head->next;
		while (cur)
		{
			Node24* next = cur->next;
			cur->next = pre;
			pre = cur;
			cur = next;
		}
		return pre;
	}
};


/*-------------------------------
		JianZhi 25
--------------------------------*/
struct Node25
{
	int val;
	Node25* next;
	Node25(int value, Node25* node)
	{
		val = value;
		next = node;
	}
};
class JianZhi25
{
public:
	Node25* MergeLists(Node25* list1, Node25* list2)
	{
		Node25* head = new Node25(0, NULL);
		Node25* p1 = list1;
		Node25* p2 = list2;

		Node25* cur = head;
		while (p1 || p2)
		{
			if (p1 && p2)
			{ 
				cur->next = p1->val < p2->val ? p1 : p2;
				p1 = p1->val < p2->val ? p1->next : p1;
				p2 = p2->val < p1->val ? p2->next : p2;
			}
			else if (!p1)
			{
				cur->next = p2;
				p2 = p2->next;
			}
			else
			{
				cur->next = p1;
				p1 = p1->next;
			}
			cur = cur->next;
		}
		return head->next;
	}
};


/*-------------------------------
		JianZhi 26
--------------------------------*/
struct BTNode26
{
	int val;
	BTNode26* left;
	BTNode26* right;
	BTNode26(int v, BTNode26* l, BTNode26* r)
	{
		val = v;
		left = l;
		right = r;
	}
};
class JianZhi26
{
public:
	bool Check(BTNode26* big, BTNode26* small)
	{
		stack<BTNode26*> S;
		BTNode26* cur = big;
		while (!S.empty() || cur)
		{
			while (cur)
			{
				S.push(cur);
				cur = cur->left;
			}
			
			if (!S.empty())
			{
				cur = S.top();
				S.pop();
				if (process(cur, small))
					return true;
				cur = cur->right;
			}
		}
		return false;
	}
private:
	bool process(BTNode26* big, BTNode26* small)
	{
		if (!small)
			return true;
		if (!big && small)
			return false;
		if (big->val != small->val)
			return false;

		return process(big->left, small->left) && process(big->right, small->right);
	}
};


/*-------------------------------
		JianZhi 27
--------------------------------*/
struct BTNode27
{
	int val;
	BTNode27* left;
	BTNode27* right;
	BTNode27(int v, BTNode27* l, BTNode27* r)
	{
		val = v;
		left = l;
		right = r;
	}
};
class JianZhi27
{
public:
	void process(BTNode* root)
	{
		if (!root)
			return;
		if (!root->left && !root->right)
			return;

		BTNode* tmp = root->left;
		root->left = root->right;
		root->right = tmp;
		process(root->left);
		process(root->right);
	}
};


/*-------------------------------
		JianZhi 28
--------------------------------*/
class JianZhi28
{
public:
	bool isSymmetrical(BTNode* root)
	{
		if (!root)
			return true;
		return process(root->left, root->right);
	}
private:
	bool process(BTNode* root1, BTNode* root2)
	{
		if (!root1 && !root2)
			return true;
		if (!root1 || !root2)
			return false;
		if (root1->val != root2->val)
			return false;

		return process(root1->left, root2->right) &&
			process(root1->right, root2->left);
	}
};


/*-------------------------------
		JianZhi 29
--------------------------------*/
class JianZhi29
{
public:
	void printMatrix(vector<vector<int>>& input)
	{
		int offset = 0;
		while (offset <= input[0].size() - 1 - offset && offset <= input.size() - 1 - offset)
		{
			printSquare(input, offset, offset, input.size() - 1 - offset, input[0].size() - 1 - offset);
			offset++;
		}
	}
private:
	void printSquare(vector<vector<int>>& input, int h1, int w1, int h2, int w2)
	{
		/*一行*/
		if (h1 == h2)
		{
			for (int cur = w1; cur <= w2; cur++)
				cout << input[h1][cur]<<endl;
			return;
		}
		/*一列*/
		if (w1 == w2)
		{
			for (int cur = h1; cur <= h2; cur++)
				cout << input[cur][w1]<<endl;
			return;
		}
		/*一圈*/
		int curh = h1;
		int curw = w1;
		while (!(curh == h1 + 1 && curw == w1))
		{
			cout << input[curh][curw] << endl;
			if (curh == h1 && curw < w2)
				curw++;
			else if (curw == w2 && curh < h2)
				curh++;
			else if (curh == h2 && curw > w1)
				curw--;
			else
				curh--;
		}
		cout<<input[curh][curw]<<endl;
	} 
};


/*-------------------------------
		JianZhi 30
--------------------------------*/
template<typename T>
struct myStack
{
public:
	void push(T newItem)
	{
		Val.push(newItem);
		Min.push(!Val.empty() && Val.top() < newItem ? Val.top() : newItem);
	}
	T pop()
	{
		T top = Val.top();
		Val.pop();
		Min.pop();
		return top;
	}
	T min()
	{
		return Min.top();
	}
private:
	stack<T> Val;
	stack<T> Min;
};


/*-------------------------------
		JianZhi 31
--------------------------------*/
class JianZhi31
{
public:
	bool isPossible(vector<int>& pushOrder, vector<int>& popOrder)
	{
		stack<int> S;
		int pPush = 0;
		int pPop = 0;
		while (pPop < popOrder.size())
		{
			if (pPush<pushOrder.size() && pushOrder[pPush] == popOrder[pPop])//入栈之后马上出栈
			{
				pPush++;
				pPop++;
			}
			else if (!S.empty() && S.top() == popOrder[pPop])//出栈
			{
				S.pop();
				pPop++;
			}
			else if (pPush < pushOrder.size())//入栈
			{
				S.push(pushOrder[pPush++]);
			}
			else//所有数已全部入栈，栈点元素不符合
				return false;
		}
		return true;
	}
};


/*-------------------------------
		JianZhi 32_3
--------------------------------*/
class JianZhi32_3
{
public:
	void ZOrder(BTNode* root)
	{
		if (!root)
			return;
		bool flag = true; //偶数行true，奇数行false
		stack<BTNode*> odd;
		stack<BTNode*> even;
		even.push(root);
		int curLevel = 1;
		int nextLevel = 0;
		BTNode* cur = NULL;
		stack<BTNode*>* source;
		stack<BTNode*>* target;
		while (!odd.empty() || !even.empty())
		{
			if (flag)
			{
				source = &even;
				target = &odd;
			}
			else
			{
				source = &odd;
				target = &even;
			}

			if (curLevel > 0)
			{
				cur = source->top();
				source->pop();
				cout << cur->val << " ";
				curLevel--;
				if (flag)
				{
					if (cur->left)
					{
						target->push(cur->left);
						nextLevel++;
					}
					if (cur->right)
					{
						target->push(cur->right);
						nextLevel++;
					}
				}
				else
				{
					if (cur->right)
					{
						target->push(cur->right);
						nextLevel++;
					}
					if (cur->left)
					{
						target->push(cur->left);
						nextLevel++;
					}
				}
			}
			else
			{
				curLevel = nextLevel;
				nextLevel = 0;
				flag = !flag;
				cout << endl;
			}
		}
	}
};


/*-------------------------------
		JianZhi 33
--------------------------------*/
class JianZhi33
{
public:
	bool isPostOrder(vector<int>& input)
	{
		return process(input, 0, input.size() - 1);
	}
private:
	bool process(vector<int>& input, int start, int end)
	{
		if (start == end)
			return true;
		if (start > end)
			return true;

		bool flag = true;
		int right = end; //右子树的起始
		for (int i = start; i < end; i++)
		{
			if (input[i] == input[end])
				return false;
			else if (flag)
			{
				if (input[i] > input[end])
				{
					flag = false;
					right = i;
				}
			}
			else
			{
				if (input[i] < input[end])
					return false;
			}
		}
		return process(input, start, right - 1) && process(input, right, end - 1);
	}
};


/*-------------------------------
		JianZhi 34
--------------------------------*/
class JianZhi34
{
public:
	vector<vector<int>> findPath(BTNode* root, int target)
	{
		vector<vector<int>> res;
		vector<int> cur;
		if (!root && target == 0)
			return res;
		cur.push_back(root->val);
		process(res, cur, root->val, root, target);
		return res;
	}
private:
	void process(vector<vector<int>>& res, vector<int>& cur, int sum, BTNode* root, int target)
	{
		if (!root->left && !root->right)
			if (target == sum)
				res.push_back(cur);

		if (root->left)
		{
			cur.push_back(root->left->val);
			process(res, cur, sum + root->left->val, root->left, target);
			cur.pop_back();
		}
		if (root->right)
		{
			cur.push_back(root->right->val);
			process(res, cur, sum + root->right->val, root->right, target);
			cur.pop_back();
		}
	}
};


/*-------------------------------
		JianZhi 35
--------------------------------*/
struct Node35
{
	int val;
	Node35* next;
	Node35* sibling;
	Node35(int value, Node35* n, Node35* s)
	{
		val = value;
		next = n;
		sibling = s;
	}
};
class JianZhi35
{
public:
	Node35* CopyNode(Node35* root)
	{
		if (!root)
			return root;
		
		Node35* cur = root;
		while (cur)
		{
			cur->next = new Node35(cur->val, cur->next, NULL);
			cur = cur->next->next;
		}

		cur = root;
		while (cur)
		{
			cur->next->sibling = cur->sibling ? cur->sibling->next : NULL;
			cur = cur->next->next;
		}

		cur = root;
		Node35* newRoot = root->next;
		Node35* res = newRoot;
		while (cur)
		{
			cur->next = cur->next->next;
			cur = cur->next;
			newRoot->next = newRoot->next ? newRoot->next->next : NULL;
			newRoot = newRoot->next;
		}
		return res;
	}

};


/*-------------------------------
		JianZhi 36
--------------------------------*/
class JianZhi36
{
public:
	BTNode* makeBiList(BTNode* root)
	{
		BTNode* cur = root;
		BTNode* pre = NULL;
		BTNode* newRoot = NULL;
		stack<BTNode*> S;
		while (!S.empty() || cur)
		{
			while (cur)
			{
				S.push(cur);
				cur = cur->left;
			}
			if (!S.empty())
			{
				cur = S.top();
				S.pop();
				if (pre)
				{
					pre->right = cur;
					cur->left = pre;
				}
				else
				{
					newRoot = cur;
				}
				pre = cur;
				cur = cur->right;
			}
		}
		return newRoot;
	}
};


/*-------------------------------
		JianZhi 37
--------------------------------*/
class JianZhi37
{
public:
	BTNode* deserialize(string& str)
	{
		BTNode* root = NULL;
		stringstream ss(str);
		_deserialize(ss, &root);
		return root;
	}
	string serialize(BTNode* root)
	{
		string str;
		if (!root)
			return str;
		_serialize(root, str);
		str.pop_back();
		return str;
	}
private:
	bool getNum(istream& ss, int& num)//从流中解析下一个数并存于引用参数num中，找到数字返回true，找到$返回false
	{
		num = 0;
		char c;
		while (ss>>c)
		{
			if (c == '$')
			{
				ss >> c;
				return false;
			}
			else if (c == ',')
				break;
			else
				num = c - '0' + 10 * num;
		}
		return true;
	}
	void _deserialize(istream& ss, BTNode** root)
	{
		int num;
		if (!getNum(ss, num))
			*root = NULL;
		else
		{
			*root = new BTNode(num, NULL, NULL);
			_deserialize(ss, &(*root)->left);
			_deserialize(ss, &(*root)->right);
		}
	}
	void _serialize(BTNode* root, string& str)
	{
		if (!root)
		{
			str.push_back('$');
			str.push_back(',');
			return;
		}
		
		str.push_back(root->val + '0');
		str.push_back(',');
		_serialize(root->left, str);
		_serialize(root->right, str);
	}
};


/*-------------------------------
		JianZhi 38
--------------------------------*/
class JianZhi38
{
public:
	void permutation(string& str)
	{
		process(str, 0);
	}
private:
	void process(string& str, int cur)
	{
		if (cur == str.size() - 1)
		{
			cout << str << endl;
			return;
		}

		for (int i = cur; i < str.size(); i++)
		{
			swap(str[i], str[cur]);
			process(str, cur + 1);
			swap(str[i], str[cur]);
		}
	}
};


/*-------------------------------
		N皇后问题
--------------------------------*/
class NQueen
{
public:
	vector<vector<int>> nQueen(int n)
	{
		vector<vector<int>> res;
		vector<int> q(n, -1);//代表皇后的位置，q[i] = j表示第i行第j列为皇后。
		process(res, q, 0);
		return res;
	}

private:
	bool canPlace(vector<int>& q, int row, int col)
	{
		for (int i = 0; i < row; i++)
		{
			if (q[i] == col || abs(i - row) == abs(q[i] - col))
				return false;
		}
		return true;
	}
	void process(vector<vector<int>>& res, vector<int>& q, int row)
	{
		if (row == q.size())
		{
			res.push_back(q);
		}
			
		for (int i = 0; i < q.size(); i++)
		{
			if (canPlace(q, row, i))
			{
				q[row] = i;
				process(res, q, row + 1);
				q[row] = -1; //回溯，但其实可以省略，canPlace并不检查当前行
			}
		}
		
	}
};


/*-------------------------------
		JianZhi 39
--------------------------------*/
class JianZhi39
{
public:
	int find(vector<int>& input)
	{
		int left = 0;
		int right = input.size() - 1;
		int mid = (left + right) >> 1;
		pair<int, int> interval = partition(input, left, right);
		while (input[mid] != input[interval.first+1])
		{
			if (interval.first+1 > mid)
				right = interval.first;
			else
				left = interval.second;
			interval = partition(input, left, right);
		}
		return input[mid];
	}

private:
	pair<int, int> partition(vector<int>& input, int start, int end)
	{
		pair<int, int> interval(start-1, end+1);
		int cur = start;
		int target = input[end];
		while (cur < interval.second)
		{
			if (input[cur] < target)
				swap(input[cur++], input[++interval.first]);
			else if (input[cur] == target)
				cur++;
			else
				swap(input[cur], input[--interval.second]);
		}
		return interval;
	}
};


/*-------------------------------
		JianZhi 40
--------------------------------*/
class JianZhi40
{
public:
	void smallestK(vector<int>& input, int k)
	{
		int restk = k;
		int start = 0; 
		int end = input.size() - 1;
		while (restk > 0)
		{
			if (end - start + 1 < k)
			{
				cout << "没有k个数";
				return;
			}
			pair<int, int> interval = partition(input, start, end);
			if (interval.first-start+2 <= restk) //把小于等于区不够（或数量刚好），先全算上，再去大于区取不足部分
			{
				restk -= (interval.first - start + 2);
				start = interval.second;
			}
			else//小于等于区太多了，把大于等于去都扔了
			{
				end = interval.first;
			}
		}
		for (int i = 0; i < k; i++)
			cout << input[i] << endl;
	}
private:
	pair<int, int> partition(vector<int>& input, int start, int end)
	{
		pair<int, int> interval(start - 1, end + 1);
		int target = input[end];
		int cur = start;
		while (cur < interval.second)
		{
			if (input[cur] < target)
				swap(input[cur++], input[++interval.first]);
			else if (input[cur] == target)
				cur++;
			else
				swap(input[cur], input[--interval.second]);
		}
		return interval;
	}
};


/*-------------------------------
		JianZhi 41
--------------------------------*/
class JianZhi41
{
public:
	double getMiddle(vector<int>& input)
	{
		priority_queue<int, vector<int>, less<int>> h1;
		priority_queue<int, vector<int>, greater<int>> h2;
		
		for (int i = 0; i < input.size(); i++)
		{
			if (h1.empty() || input[i] <= h1.top())
				h1.push(input[i]);
			else
				h2.push(input[i]);
			balance(h1, h2);
		}
		double res;
		if ((input.size() & 1) == 1)
			res = h1.size() > h2.size() ? h1.top() : h2.top();
		else
			res = (h1.top() + h2.top()) / 2.0;
		return res;
	}
private:
	void balance(priority_queue<int, vector<int>, less<int>>& h1, priority_queue<int, vector<int>, greater<int>>& h2)
	{
		if (h1.size() - h2.size() == 2)
		{
			int val = h1.top();
			h1.pop();
			h2.push(val);
		}
		else if (h2.size() - h1.size() == 2)
		{
			int val = h2.top();
			h2.pop();
			h1.push(val);
		}
	}
};


/*-------------------------------
		排序算法总结
--------------------------------*/
class mySort
{
public:
	void BubbleSort(vector<int>& m)
	{
		for (int i = 0; i < m.size(); i++)
			for (int j = m.size() - 1; j > i; j--)
				if (m[i] > m[j])
					swap(m[i], m[j]);
	}
	void SelectSort(vector<int>& m)
	{
		for (int i = 0; i < m.size(); i++)
			for (int j = i + 1; j < m.size(); j++)
				if (m[j] < m[i])
					swap(m[i], m[j]);
	}
	void InsertSort(vector<int>& m)
	{
		for (int i = 1; i < m.size(); i++) //从1开始是因为初始默认m[0]是已排序数
		{
			int tmp = m[i];
			int j = i - 1;
			for (; j >= 0; j--)
				if (m[j] > m[i])
					m[j + 1] = m[j];
				else
					break;

			m[j + 1] = tmp;
		}
	}
	void ShellSort(vector<int>& m)
	{
		for (int step = m.size() / 2; step > 0; step /= 2) //初始增量为序列长度的一半，每次减半
			for (int i = step; i < m.size(); i++)
			{
				int tmp = m[i];
				int j = i - step;
				for (; j > 0; j -= step)
				{
					if (tmp < m[j])
						m[j + step] = m[j];
					else
						break;
				}
				m[j + step] = tmp;
			}
	}
	void MergeSort(vector<int>& m)
	{
		_MergeSortProcess(m, 0, m.size() - 1);
	}
	void HeapSort(vector<int>& m)
	{
		int heapSize = 1;
		for (int i = 1; i < m.size(); i++)
			_HeapInsert(m, heapSize);
		for (int i = 1; i < m.size(); i++)
			_Heapify(m, heapSize);
	}
	void QuickSort(vector<int>& m)
	{
		_QuickSortProcess(m, 0, m.size() - 1);
	}

private:
	/*MergeSort*/
	void _Merge(vector<int>& m, int left, int mid, int right)
	{
		vector<int> tmp;
		int curL = left;
		int curR = mid + 1; 
		while (curL != mid + 1 || curR != right + 1)
		{
			if (curL == mid + 1)
				tmp.push_back(m[curR++]);
			else if (curR == right + 1)
				tmp.push_back(m[curL++]);
			else
				tmp.push_back(m[curL] <= m[curR] ? m[curL++] : m[curR++]); //等号是为了保证稳定性
		}
		for (int i = 0; i < tmp.size(); i++)
			m[left + i] = tmp[i];
	}
	void _MergeSortProcess(vector<int>& m, int left, int right)
	{
		if (left >= right)
			return;

		int mid = (left + right) >> 1;
		_MergeSortProcess(m, left, mid);
		_MergeSortProcess(m, mid + 1, right);
		_Merge(m, left, mid, right);
	}
	/*HeapSort*/
	void _HeapInsert(vector<int>& m, int& heapSize)
	{
		int curIdx = heapSize;
		int parentIdx = (heapSize - 1) >> 1;
		while (m[parentIdx] < m[curIdx])
		{
			swap(m[parentIdx], m[curIdx]);
			curIdx = parentIdx;
			parentIdx = (curIdx - 1) / 2;
		}
		heapSize++;
	}
	void _Heapify(vector<int>& m, int& heapSize)
	{
		swap(m[0], m[heapSize - 1]);
		heapSize--;
		int cur = 0;
		int leftChildIdx = 1;
		while (leftChildIdx < heapSize)
		{
			int largerChildIdx = leftChildIdx;
			largerChildIdx = leftChildIdx + 1 < heapSize && m[leftChildIdx + 1] > m[leftChildIdx] ? leftChildIdx + 1 : leftChildIdx;
			if (m[largerChildIdx] > m[cur])
			{
				swap(m[largerChildIdx], m[cur]);
				cur = largerChildIdx;
				leftChildIdx = 2 * cur + 1;
			}
			else
				break;
		}
	}
	/*QuickSort*/
	void _QuickSortProcess(vector<int>& m, int left, int right)
	{
		pair<int, int> interval = _Partition(m, left, right);
		if (interval.first > left)
			_QuickSortProcess(m, left, interval.first);
		if (interval.second < right)
			_QuickSortProcess(m, interval.second, right);
	}
	pair<int, int> _Partition(vector<int>& m, int left, int right)
	{
		pair<int, int> interval{ left - 1, right+1}; //(小于区的右边界， 大于区的左边界)
		int target = m[right];
		int cur = left;
		while (cur < interval.second)
		{
			if (m[cur] < target)
				swap(m[++interval.first], m[cur++]);
			else if (m[cur] == target)
				cur++;
			else 
				swap(m[--interval.second], m[cur]);
		}
		return interval;
	}

};


/*-------------------------------
		JianZhi 42
--------------------------------*/
class JianZhi42
{
public:
	/*双指针*/
	int GetMaxSubset(vector<int>& input)
	{
		int _max = 0;
		for (int e : input)
			_max += e;
		int cur = _max;
		int left = 0; 
		int right = input.size()-1;
		while (left <= right)
		{
			if (input[left] >= input[right])
				_max = max(_max, cur-=input[right--]);
			else
				_max = max(_max, cur-=input[left++]);
		}
		return _max;
	}
	/*一遍遍历硬怼，个人认为规律不容易想到也容易出错*/
	int GetMaxSubset2(vector<int>& input)
	{
		int _max = 0;
		int curMax = input[0];
		for (int i = 1; i < input.size(); i++)
		{
			if (curMax < 0)
				curMax = input[i];
			else
				curMax += input[i];
			_max = max(curMax, _max);
		}
		return _max;
	}
	/*递归*/
	int GetMaxSubset3(vector<int>& input)
	{
		int _max = 0;
		for (int i = 0; i < input.size(); i++)
		{
			int cur = process(input, i);
			_max = max(_max, cur);
		}
		return _max;
	}
	/*dp版本，硬怼版本其实是优化后的dp，但是真的难以想到*/
	int GetMaxSubset4(vector<int>& input)
	{
		vector<int> dp(input.size(), 0);
		dp[0] = input[0];
		int _max = max(0, dp[0]);
		for (int i = 1; i < dp.size(); i++)
		{
			dp[i] = input[i] + (dp[i - 1] < 0 ? 0 : dp[i - 1]);
			_max = max(_max, dp[i]);
		}
		return _max;
	}
private:
	int process(vector<int>& input, int right)
	{
		if (right == 0)
			return input[0];
		int last = process(input, right - 1);
		return input[right] + (last < 0 ? 0 : last);
	}
};


/*-------------------------------
华为笔试1: 需要n个钉子，店里有4只
装和9只装两种，问最少需要买几包。
--------------------------------*/
class HuaWei_19_9_4_1
{
public:
	/*递归*/
	int process(int n)
	{
		if (n < 0)
			return INT_MAX;
		if (n == 0)
			return 0;

		int _min = min(process(n - 9), process(n - 4));
		return _min == INT_MAX ? INT_MAX : (1+_min) ;
	}
	/*dp*/
	int getMin(int n)
	{
		vector<int> dp(n+1, -1);
		dp[0] = 0;
		if (n <= 3)
			return dp[n];
		dp[4] = 1;
		if (n <= 7)
			return dp[n];
		dp[8] = 2;
		if (n < 9)
			return dp[n];
		dp[9] = 1;
		if (n <= 9)
			return dp[n];

		for (int i = 10; i <= n; i++)
		{
			if (dp[i - 4] == -1)
				dp[i] = dp[i - 9];
			else if (dp[i - 9] == -1)
				dp[i] = dp[i - 4];
			else
				dp[i] = min(dp[i - 4], dp[i - 9]);

			if (dp[i] != -1)
				dp[i]++;
		}

		return dp[n];
	}
};


/*-------------------------------
		JianZhi 43
--------------------------------*/
class JianZhi43
{
public:
	int TimesOf1(int n)
	{
		if (n <= 0)
			return 0;
		char cStr[50];
		sprintf_s(cStr, "%d", n);
		string N(cStr);
		return process(N);
	}
private:
	int process(string N)
	{
		if (N.size() == 1)
			return 1;
		
		int sum = 0;
		/*计算最高位1出现次数*/
		if (N[0] == '1')//最高位为1
		{
			for (int i = 1; i <= N.size()-1; i++)
				sum = 10 * sum + (N[i]-'0');
			sum += 1;
		}
		else //最高位大于1
			sum = pow(10, N.size()-1);
		/*计算剩下位1出现次数*/
		sum += (N[0] - '0') * pow(10, N.size()-2) * (N.size()-1);
		
		return sum + process(N.substr(1));
	}
};


/*-------------------------------
盛趣笔试1：字符串由'.'或'_'隔开，
现反转字符串，'.'后面的整体反转，
'_'后面的整体反转且内部反转。第一个
单词内部不反转。
例：hello.world_haha => ahah_world.hello
--------------------------------*/
class ShengQu_19_9_9_1
{
public:
	string Reverse()
	{
		string input("seek.to");
		string res = reverseWords(input);
		return res;
	}
private:
	string reverseWords(string& input)
	{
		stack<char> S;
		string res(input);
		int pRes = res.size() - 1;
		int pInput = 0;
		while (pInput < input.size())
		{
			if (input[pInput] != '.' && input[pInput] != '_')
				S.push(input[pInput++]);
			else if (input[pInput] == '.')
			{
				while (!S.empty())
				{
					res[pRes--] = S.top();
					S.pop();
				}
				res[pRes--] = '.';
				pInput++;
			}
			else
			{
				while (!S.empty())
				{
					res[pRes--] = S.top();
					S.pop();
				}
				res[pRes--] = '_';
				pInput++;
				while (input[pInput] != '.' && pInput != res.size())
				{
					res[pRes--] = input[pInput++];
				}
			}
		}
		while (!S.empty())
		{
			res[pRes--] = S.top();
			S.pop();
		}

		return res;
	}
};


/*-------------------------------
盛趣笔试2：不用任何循环、递归，输
入n，打印0到n。
--------------------------------*/
class CAssistant
{
public:
	CAssistant()
	{
		cout << cnt++;
	}
private:
	static int cnt;
};
int CAssistant::cnt = 0;
class ShengQu_19_9_9_2
{
public:
	void go()
	{
		int cnt;
		cin >> cnt;
		CAssistant* arr = new CAssistant[cnt + 1];
	}
};


/*-------------------------------
二维数组，左上到右下共有多少种走法
--------------------------------*/
class LeftUp2RightDown
{
public:
	int Recursive(int row, int col)
	{
		return process(row, col, 0, 0);
	}

	int DP(int row, int col)
	{
		vector<vector<int>> dp(row, vector<int>(col));
		dp[row - 1][col - 1] = 1;

		for (int i = row - 2; i >= 0; i--)
			dp[i][col - 1] = dp[i + 1][col - 1];

		for (int i = col - 2; i >= 0; i--)
			dp[row - 1][i] = dp[row - 1][i+1];

		for (int i = row - 2; i >= 0; i--)
			for (int j = col - 2; j >= 0; j--)
				dp[i][j] = dp[i + 1][j] + dp[i][j + 1];

		return dp[0][0];
	}
private:
	int process(const int& row, const int& col, int i, int j)
	{
		if (i == row - 1 && j == col - 1)
			return 1;

		if (i == row - 1)
			return process(row, col, i, j + 1);

		if (j == col - 1)
			return process(row, col, i + 1, j);

		return process(row, col, i, j + 1) + process(row, col, i + 1, j);
	}
};


/*-------------------------------
找到一个数组中，出现次数超过一半的数
--------------------------------*/
class FindMoreThanHalf
{
public:
	int Search(vector<int>& input) 
	{
		int left = 0;
		int right = input.size() - 1;
		int mid = (left + right) / 2; 
		pair<int, int> interval = partition(input, 0, input.size() - 1);
		while (input[interval.first + 1] != input[mid])
		{
			if (interval.first + 1 > mid)
				right = interval.first;
			else
				left = interval.second;

			interval = partition(input, left, right);
		}
		return input[mid];
	}

private:
	pair<int, int> partition(vector<int>& input, int left, int right)
	{
		pair<int, int> interval{ left - 1, right + 1 };
		int cur = left;
		int target = input.back();
		while (cur < interval.second)
		{
			if (input[cur] < target)
				swap(input[cur++], input[++interval.first]);
			else if (input[cur] == target)
				cur++;
			else
				swap(input[cur], input[--interval.second]);
		}
		return interval;
	}
};


/*-------------------------------
字节面试：开根
--------------------------------*/
class ZiJie1
{
public:
	float sqrt(float target)
	{
		float left = 0;
		float right = target;
		float mid = (left + right) / 2;
		int flag = isEqual(mid, target);
		while (flag != 0)
		{
			if (flag == -1)
				left = mid;
			if (flag == 1)
				right = mid;
			mid = (left + right) / 2;
			flag = isEqual(mid, target);
		}
		return mid;
	}
private:
	int isEqual(float cur, float target)
	{
		if (cur * cur - target > 1e-6)
			return 1;
		else if (target - cur * cur > 1e-6)
			return -1;
		return 0;
	}
};


/*-------------------------------
字节面试：数组中两个数出现1次，
其余数出现两次，找到两个出现1次的数
--------------------------------*/
class ZiJie2
{
public:
	void findOnceNum(const vector<int>& input)
	{
		int XOR = input[0];
		for (int i = 1; i<input.size(); i++)
		{
			XOR = XOR ^ input[i];
		}

		int test = 1;
		while ((XOR & test) != 1)
		{
			test << 1; 
		}

		bool first0found = false;
		bool first1found = false;
		int XOR0;
		int XOR1;
		for (int num : input)
		{
			if ((num & test) == 0)
			{
				if (!first0found)
				{
					XOR0 = num;
					first0found = true;
				}
				else
				{
					XOR0 = XOR0 ^ num;			
				}
			}
			else
			{
				if (!first1found)
				{
					XOR1 = num;
					first1found = true;
				}
				else
				{
					XOR1 = XOR1 ^ num;
				}
			}
		}

		cout << XOR0 << " " << XOR1;
	}
};


/*-------------------------------
JianZhi44:数字序列中某一位的数字
序列为012345678910111213......
--------------------------------*/
class JianZhi44
{
public:
	void find(int n)
	{
		if (n < 10)
			cout<<n;
		n -= 10;
		int num = 2; //几位数
		int place = 9 * num * pow(10, num - 1);
		while (n >= place)
		{
			n -= place;
			num++;
			place = 9 * num * pow(10, num - 1);
		}

		int cnt = n / num; //num位数中的第几个
		int idx = n % num; //cnt中的第几位
		idx = num - idx - 1; //cnt中右数第几位

		n = pow(10, num-1) + cnt;
		
		for (int i = 0; i < idx; i++)
			n /= 10;

		cout << n % 10;
	}
};


/*-------------------------------
JianZhi45:把数组排成最小的数
例如：{3,32,321} => 321323
--------------------------------*/
class JianZhi45
{
public:
	void getMin(const vector<int>& input)
	{
		vector<string> strs;
		char tmp[10];
		string str;
		for (int n : input)
		{			
			sprintf_s(tmp, "%d", n);
			str = tmp;
			strs.push_back(str);
		}
		sort(strs.begin(), strs.end(), cmp);
		string res = "";
		for (string s : strs)
			res += s;
		cout << res;
	}
private:
	static bool cmp(string str1, string str2)
	{
		return str1 + str2 < str2 + str1;
	}
};


/*-------------------------------
JianZhi46:把数字翻译成字符串,计算
可能的翻译个数
0 -> 'a', 1 -> 'b'...25 -> 'z'
--------------------------------*/
class JianZhi46
{
public:
	void count(int n)
	{
		char tmp[50];
		sprintf_s(tmp, "%d", n);
		string str(tmp);
		cout<<process(str, 0);
	}
	void DP(int n)
	{
		char tmp[50];
		sprintf_s(tmp, "%d", n);
		string str(tmp);
		vector<int> dp(str.size() + 1, 0);
		dp[dp.size() - 1] = 1;
		for (int cur = dp.size() - 2; cur >= 0; cur--)
		{
			dp[cur] = dp[cur + 1];
			if (cur != str.size() - 1 && (str[cur] == '2' && str[cur + 1] <= '5') || str[cur] == '1')
				dp[cur] += dp[cur + 2];
		}
		cout << dp[0];
	}
private:
	int process(string& str, int cur)
	{
		if (cur == str.size())
			return 1;

		int cnt = process(str, cur + 1);
		if (cur != str.size() - 1 && (str[cur] == '2' && str[cur + 1] <= '5') || str[cur] == '1')
			cnt += process(str, cur + 2);
		return cnt;
	}
};


/*-------------------------------
JianZhi47:礼物的最大价值
--------------------------------*/
class JianZhi47
{
public:
	void biggest(const vector<vector<int>>& m)
	{
		int profit = process(m, 0, 0);
		cout << profit;
	}
	void DP(const vector<vector<int>>& m)
	{
		int row = m.size();
		int col = m[0].size();
		vector<vector<int>> dp(row, vector<int>(col));
		dp[row - 1][col - 1] = m[row - 1][col - 1];
		for (int i = col - 2; i >= 0; i--)
			dp[row - 1][i] = dp[row - 1][i + 1] + m[row-1][i];
		for (int i = row - 2; i >= 0; i--)
			dp[i][col - 1] = dp[i + 1][col-1] + m[i][col-1];

		for (int i = row - 2; i >= 0; i--)
			for (int j = col - 2; j >= 0; j--)
				dp[i][j] = max(dp[i + 1][j], dp[i][j + 1]) + m[i][j];

		cout << dp[0][0];
	}
	void DP2(const vector<vector<int>>& m)
	{
		int row = m.size();
		int col = m[0].size();
		vector<int> dp(col);
		dp[col - 1] = m[row - 1][col - 1];
		for (int i = col - 2; i >= 0; i--)
			dp[i] = dp[i + 1] + m[row - 1][i];

		for (int i = row-2; i>=0; i--)
			for (int j = col-1; j >= 0; j--)
			{
				if (j == col - 1)
					dp[j] += m[i][j];
				else
					dp[j] = max(dp[j], dp[j + 1]) + m[i][j];
			}
		cout << dp[0];
	}
private:
	int process(const vector<vector<int>>& m, int i, int j)
	{
		if (i == m.size() - 1 && j == m[0].size() - 1)
			return m[i][j];
		
		if (i == m.size() - 1)
			return process(m, i, j + 1) + m[i][j];

		if (j == m[0].size() - 1)
			return process(m, i + 1, j) + m[i][j];

		return max(process(m, i, j + 1), process(m, i + 1, j)) + m[i][j];
	}
};


/*-------------------------------
JianZhi48:最长不含重复字符的子字符串
--------------------------------*/
class JianZhi48
{
public:
	void longest(const string& str)//贪婪
	{
		int idx[26];
		for (int i = 0; i < 26; i++)
			idx[i] = -1;
		int left = 0;
		int right = 0;
		int maxLength = 0;
		int curLength = 0;
		int bestLeft = 0;
		int bestRight = 0;
		while (right < str.size())
		{
			if (idx[str[right] - 'a'] == -1 || right - idx[str[right] - 'a'] > right - left)
			{
				idx[str[right] - 'a'] = right;
				right++;
			}
			else
			{
				curLength = right - left;
				if (curLength > maxLength)
				{
					maxLength = curLength;
					bestLeft = left;
					bestRight = right - 1;
				}
				left = idx[str[right] - 'a'] + 1;
				idx[str[right] - 'a'] = right;
				right++;
			}
		}
		if (curLength > maxLength)
		{
			maxLength = curLength;
			bestLeft = left;
			bestRight = right;
		}
		cout << str.substr(bestLeft, bestRight - bestLeft + 1);
	}
	void recursive(const string& str)
	{
		maxLength = 1;
		process(str, str.size() - 1);
		cout << bestStr;
	}
	void DP(const string& str)
	{
		vector<int> dp(str.size());
		dp[0] = 1;
		bestStr = str[0];
		maxLength = 1;

		for (int i = 1; i < dp.size(); i++)
		{
			int j = 1;
			for (; j <= dp[i - 1]; j++)
			{
				if (str[i] == str[i - j])
					break;
			}
			dp[i] = j;
			if (j > maxLength)
			{
				maxLength = j;
				bestStr = str.substr(i - j + 1, j);
			}
		}
		cout << bestStr;
	}
private:
	int process(const string& str, int cur)
	{
		if (cur == 0)
			return 1;
		
		int cnt = process(str, cur - 1);
		int i = 1;
		for (; i <= cnt; i++)
		{
			if (str[cur] == str[cur - i])
				break;
		}
		if (i > maxLength)
		{
			maxLength = i;
			bestStr = str.substr(cur-i+1, i);
		}
		return i;
	}
	int maxLength;
	string bestStr;
};


/*-------------------------------
JianZhi49:丑数
--------------------------------*/
class JianZhi49
{
public:
	void find()
	{
		vector<int> ugly{ 1, 2, 3, 4, 5 };
		int t2 = 2;
		int t3 = 1;
		int t5 = 1;
		while (ugly.size() != 1500)
		{
			int _min = min(ugly[t2] * 2, ugly[t3]*3);
			_min = min(_min, ugly[t5] * 5);
			ugly.push_back(_min);

			while (ugly[t2] * 2 <= _min)
				t2++;
			while (ugly[t3] * 3 <= _min)
				t3++;
			while (ugly[t5] * 5 <= _min)
				t5++;
		}
		cout << ugly.back();
	}
private:
	
	
};


/*-------------------------------
JianZhi50:第一个只出现一次的字符
--------------------------------*/
class JianZhi50
{
public:
	//略
};


/*-------------------------------
JianZhi53:在排序数组中查找数字出现次数
--------------------------------*/
class JianZhi53
{
public:
	int find(const vector<int>& input, int target)
	{
		int left = findLeft(input, target);
		if (left == -1)
			return -1;
		int right = findRight(input, target);
		return right - left + 1;
	}

private:
	int findLeft(const vector<int>& input, const int target)
	{
		int left = 0;
		int right = input.size() - 1;
		int mid;
		while (left <= right)
		{
			mid = (left + right) / 2;
			if (input[mid] == target)
			{
				if (mid > 0 && input[mid - 1] != target || mid == 0)
					return mid;
				else
					right = mid - 1;
			}
			else if (input[mid] < target)
				left = mid + 1;
			else
				right = mid - 1;
		}
		return -1;
	}
	int findRight(const vector<int>& input, int target)
	{
		int left = 0;
		int right = input.size()-1;
		int mid;
		while (left <= right)
		{
			mid = (left + right) / 2;
			if (input[mid] == target)
			{
				if (mid < input.size() - 1 && input[mid + 1] != target || mid == input.size()-1)
					return mid;
				else
					left = mid + 1;
			}
			else if (input[mid] < target)
				left = mid + 1;
			else
				right = mid - 1;
		}
		return -1;
	}

};



/*-------------------------------
JianZhi53_2:0-n-1中找出唯一缺失的数
--------------------------------*/
class JianZhi53_2
{
public:
	int find(const vector<int>& input)
	{
		int left = 0;
		int right = input.size() - 1;
		int mid = 0;
		while (left <= right)
		{
			mid = (left + right) / 2;
			if (input[mid] != mid)
			{
				if (mid == 0 || mid > 0 && input[mid - 1] == mid - 1)
					return mid;
				else
					right = mid - 1;
			}
			else
				left = mid + 1;
		}
		return -1;
	}

};











/*------------------------------------------------------
在数轴上x=0的位置上有一只青蛙，它的家在x=n的位置。
现在它想从x=0的位置跳回家。他决定第一次跳跃的长度为1，
之后每一次跳跃的长度比上一次跳跃的长度大一个单位。
也就是说第i次跳跃的长度为i。每一次跳跃可以选择往左
或者往右跳。它想知道最少要经过多少次跳跃才能到达终点。
------------------------------------------------------*/
class FragGoHome
{
public:
	void jump()
	{
		int n = 3;
		int right = 0;
		int left = 0;
		int cnt = 0;
		while (right < n)
		{
			left = right;
			right += ++cnt;
		}

		int p = n - left >= right - n ? right : left;
		cnt = n - left >= right - n ? cnt : cnt - 1;
		cout << cnt + abs(p - n) * 2;
	}
};








int main()
{

	
	return 0;
}