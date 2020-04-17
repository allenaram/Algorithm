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
#include <unordered_set>


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


/*JianZhi Start*/



/*----------------------------------------
			JianZhi 3.1
 ---------------------------------------*/
class JianZhi3_1
{
public:
	/* 哈希表法 */
	vector<int> repeatedNum(vector<int>& input)
	{
		vector<int> res;
		unordered_set<int> Set;
		for (int i = 0; i < input.size(); i++)
			if (Set.find(input[i]) != Set.end())
				res.push_back(input[i]);
			else
				Set.insert(input[i]);

		return res;
	}
	/* 交换法 */
	vector<int> repeatedNum2(vector<int>& input)
	{
		vector<int> res;
		int i = 0;
		while (i < input.size())
		{
			if (i != input[i])
				if (input[i] == input[input[i]])
					res.push_back(input[i++]);
				else
					swap(input[i], input[input[i]]);
			else
				i++;
		}
		return res;
	}
};


/*----------------------------------------
			JianZhi 3.2
 ---------------------------------------*/
class JianZhi3_2
{
public:
	int count(vector<int>& input, int start, int end)
	{
		int cnt = 0;
		for (int i = 0; i < input.size(); i++)
			cnt = input[i] >= start && input[i] <= end ? cnt + 1 : cnt;
		return cnt;
	}

	int repeatedNum1(vector<int>& input)
	{
		/*核心思想：idx 1-m若不重复，最多只能有m个数*/
		int start = 1;
		int end = input.size() - 1;
		while (start <= end)
		{
			int mid = (start + end) >> 1;
			if (count(input, start, mid) > mid - start + 1)
				if (start == mid)
					return start;
				else
					end = mid;
			else
				start = mid + 1;
		}
		return start;
	}
};


/*----------------------------------------
			JianZhi 4
 ---------------------------------------*/
class JianZhi4
{
public:
	bool search(vector<vector<int>> input, int target)
	{
		int height = input.size();
		int width = input[0].size();
		int i = 0;
		int j = width - 1;
		while (true)
		{
			if (input[i][j] == target)
				return true;
			else if (input[i][j] > target)
				j--;
			else
				i++;
			if (i >= height || j < 0)
				return false;
		}
		return false;
	}
};


/*----------------------------------------
			JianZhi 5
 ---------------------------------------*/
class JianZhi5
{
public:
	void  ReplaceBlank(string& str)
	{
		int cnt = 0;
		for (int i = 0; i < str.size(); i++)
			if (str[i] == ' ')
				cnt++;

		int pOrigin = str.size() - 1;
		int pNew = str.size() + 2 * cnt - 1;
		str.append(2 * cnt, ' ');
		while (pOrigin >= 0)
		{
			if (str[pOrigin] != ' ')
				str[pNew--] = str[pOrigin--];
			else
			{
				str[pNew--] = '0';
				str[pNew--] = '2';
				str[pNew--] = '/';
				pOrigin--;
			}
		}

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



/*Leetcode start*/

/*-----------------------------------------------------
	leetcode 1
------------------------------------------------------*/
vector<int> twoSum(vector<int>& nums, int target)
{
	vector<int> answer;
	unordered_map<int, int> hashmap; //(value, index)

	for (int i = 0; i < nums.size(); i++)
	{
		unordered_map<int, int>::iterator it;
		it = hashmap.find(target - nums[i]);
		if (it == hashmap.end())
			hashmap[nums[i]] = i;
		else if (i != it->second)
		{
			answer.push_back(i);
			answer.push_back(it->second);
			break;
		}
	}
	return answer;
}


/*-----------------------------------------------------
	leetcode 2
------------------------------------------------------*/
struct ListNode
{
	int val;
	ListNode *next;
	ListNode(int x) : val(x), next(NULL) {}
};
class Solution2 {
public:
	ListNode* addTwoNumbers(ListNode* l1, ListNode* l2)
	{
		ListNode* head = new ListNode(0);
		ListNode* cur = head;
		int v1 = l1->val;
		int v2 = l2->val;
		int carry = 0;
		int val = 0;
		while (l1 || l2)
		{
			v1 = l1 ? l1->val : 0;
			v2 = l2 ? l2->val : 0;
			val = v1 + v2 + carry;
			carry = val > 9 ? 1 : 0;
			cur->next = new ListNode(val % 10);
			cur = cur->next;
			l1 = l1 ? l1->next : NULL;
			l2 = l2 ? l2->next : NULL;
		}
		if (carry > 0)
		{
			cur->next = new ListNode(carry);
		}
		return head->next;
	}
};


/*-----------------------------------------------------
	leetcode 3
------------------------------------------------------*/
class Solution3 {
public:
	int lengthOfLongestSubstring(string s) {
		int maxSubstrLen = 0;
		int Len = 0;
		int back = 0;
		unordered_set<char> hashset;
		for (int i = 0; i < s.length(); i++)
		{
			if (hashset.find(s[i]) == hashset.end())
			{
				hashset.insert(s[i]);
				Len++;
			}
			else
			{
				if (Len > maxSubstrLen)
					maxSubstrLen = Len;
				Len = 0;
				back = 0;
				hashset.clear();
				while (hashset.find(s[i - back]) == hashset.end())
				{
					hashset.insert(s[i - back]);
					Len++;
					back++;
				}
			}
		}
		if (Len > maxSubstrLen)
			maxSubstrLen = Len;
		cout << maxSubstrLen;
		return maxSubstrLen;
	}
};


/*-----------------------------------------------------
	leetcode 5
------------------------------------------------------*/
class Solution5 {
public:
	string longestPalinedrome(string s)	//动态规划版
	{
		int maxi = 0;
		int maxlen = 1;
		bool matrix[1000][1000] = { false };
		for (int i = 0; i < s.length(); i++)
			matrix[i][i] = true;
		for (int i = 0; i < s.length() - 1; i++)
		{
			if (s[i] == s[i + 1])
			{
				matrix[i][i + 1];
				maxlen = 2;
				maxi = i;
			}

		}
		for (int l = 3; l < s.length(); l++)
			for (int i = 0; i + l - 1 < s.length(); i++)
			{
				if (matrix[i + 1][i + l - 2] && s[i + 1] == s[i + l - 2])
				{
					matrix[i][i + l - 1] = true;
					if (l > maxlen)
					{
						maxi = i;
						maxlen = l;
					}
				}
			}
		return s.substr(maxi, maxlen);


	}


	string longestPalindrome1(string s) {	//暴力递归版
		int max = 1;
		int maxi = 0;
		int i, j;
		for (i = 0; i < s.length(); i++)
		{
			for (j = i; j < s.length(); j++)
				if (process(i, j, s) && (j - i + 1) > max)
				{
					max = j - i + 1;
					maxi = i;
				}
		}
		return s.substr(maxi, max);
	}

	bool process(int i, int j, string s)
	{
		if (i >= j)
			return true;
		if (i + 1 == j)
			return s[i] == s[j];
		return (process(i + 1, j - 1, s) && s[i] == s[j]);
	}
};


/*-----------------------------------------------------
	leetcode 6
------------------------------------------------------*/
class Solution6 {
public:
	string convert(string s, int numRows) {
		if (numRows == 1)
			return s;

		int len = s.length();
		vector<string> strs(min(numRows, len));
		bool flag = false;
		int i = 0;
		for (char c : s)
		{
			strs[i] += c;
			if (i == 0 || i == numRows - 1)
				flag = !flag;
			i = flag ? i + 1 : i - 1;
		}


		string fs = "";
		for (int i = 0; i < strs.size(); i++)
			fs += strs[i];
		return fs;
	}

	string convert2(string s, int numRows)
	{
		if (numRows == 1) return s;

		string ret;
		int n = s.size();
		int cycleLen = 2 * numRows - 2;

		for (int i = 0; i < numRows; i++) {
			for (int j = 0; j + i < n; j += cycleLen) {
				ret += s[j + i];
				if (i != 0 && i != numRows - 1 && j + cycleLen - i < n)
					ret += s[j + cycleLen - i];
			}
		}
		return ret;
	}
};


/*-----------------------------------------------------
	leetcode 7  注意溢出情况
------------------------------------------------------*/
class Solution7 {
public:
	int reverse(int x) {
		int rev = 0;
		while (x != 0) {
			int pop = x % 10;
			x /= 10;
			if (rev > INT_MAX / 10 || (rev == INT_MAX / 10 && pop > 7)) return 0;
			if (rev < INT_MIN / 10 || (rev == INT_MIN / 10 && pop < -8)) return 0;
			rev = rev * 10 + pop;
		}
		return rev;
	}
};


/*-----------------------------------------------------
	leetcode 429
------------------------------------------------------*/
class Node429 {
public:
	int val;
	vector<Node429*> children;

	Node429() {}

	Node429(int _val, vector<Node429*> _children) {
		val = _val;
		children = _children;
	}
};
class Solution429 {
public:
	vector<vector<int>> levelOrder(Node429* root) {
		vector<vector<int>> res;
		vector<int> res0;
		queue<Node429*> Q;
		if (!root)
			return res;
		Q.push(root);
		int cur = 1;
		int next = 0;

		Node429* c = NULL;
		while (!Q.empty())
		{
			c = Q.front();
			res0.push_back(c->val);
			Q.pop();
			cur--;
			for (int i = 0; i < c->children.size(); i++)
			{
				Q.push(c->children[i]);
				next++;
			}
			if (cur == 0)
			{
				cur = next;
				next = 0;
				res.push_back(res0);
				res0.clear();
			}
		}
		return res;
	}
};


/*-----------------------------------------------------
	leetcode 9
------------------------------------------------------*/
class Solution8 {
public:
	bool isPalindrome(int x) {
		string str;
		if (x < 0)
			return 0;
		int val = 0;
		while (x != 0)
		{
			str.push_back(x % 10);
			x /= 10;
		}
		for (int i = 0; i <= str.size() / 2; i++)
		{
			if (str[i] != str[str.size() - 1 - i])
				return false;
		}
		return true;
	}
	bool isPalindrome2(int x) {
		if (x >= 0 && x < 10)
			return true;
		if (x < 0 || x % 10 == 0)
			return false;
		int tmp = x;
		int num = 0;
		while (tmp != 0)
		{
			tmp /= 10;
			num++;
		}
		num /= 2;

		for (int i = 0; i < num; i++)
		{
			tmp = 10 * tmp + x % 10;
			x /= 10;
		}
		if (x < 10 && tmp < 10)
			return x == tmp;
		return x == tmp || x / 10 == tmp;

	}
};


/*-----------------------------------------------------
	leetcode 11
------------------------------------------------------*/
class Solution11 {
public:
	int maxArea(vector<int>& height) {
		int i = 0;
		int j = height.size() - 1;
		int maxi = i;
		int maxj = j;
		int maxV = 0;
		int V = 0;
		while (i < j)
		{
			V = min(height[i], height[j])*(j - i);
			if (V > maxV)
			{
				maxV = V;
				maxi = i;
				maxj = j;
			}
			if (height[i] < height[j])
				i++;
			else if (height[i] > height[j])
				j--;
			else
			{
				i++; j--;
			}
		}
		return maxV;
	}
};


/*-----------------------------------------------------
	leetcode 12
------------------------------------------------------*/
class Solution12 {
public:
	string intToRoman(int num) {
		string str;
		while (num > 0)
		{
			if (num >= 1000)
			{
				str += 'M';
				num -= 1000;
			}
			else if (num >= 900)
			{
				str += "CM";
				num -= 900;
			}
			else if (num >= 500)
			{
				str += 'D';
				num -= 500;
			}
			else if (num >= 400)
			{
				str += "CD";
				num -= 400;
			}
			else if (num >= 100)
			{
				str += 'C';
				num -= 100;
			}
			else if (num >= 90)
			{
				str += "XC";
				num -= 90;
			}
			else if (num >= 50)
			{
				str += 'L';
				num -= 50;
			}
			else if (num >= 40)
			{
				str += "XL";
				num -= 40;
			}
			else if (num >= 10)
			{
				str += 'X';
				num -= 10;
			}
			else if (num >= 9)
			{
				str += "IX";
				num -= 9;
			}
			else if (num >= 5)
			{
				str += 'V';
				num -= 5;
			}
			else if (num >= 4)
			{
				str += "IV";
				num -= 4;
			}
			else if (num >= 1)
			{
				str += 'I';
				num -= 1;
			}
		}
		return str;
	}
};


/*-----------------------------------------------------
	leetcode 13
------------------------------------------------------*/
class Solution13 {
public:
	int romanToInt(string s) {
		int num = 0;
		if (s.empty())
			return num;
		int i = 0;
		while (i < s.size())
		{
			if (s[i] == 'M')
			{
				num += 1000;
				i++;
			}
			else if (s[i] == 'D')
			{
				num += 500;
				i++;
			}
			else if (s[i] == 'C')
			{
				if (s[i + 1] == 'M' && i != s.size() - 1)
				{
					num += 900;
					i += 2;
				}
				else if (s[i + 1] == 'D' && i != s.size() - 1)
				{
					num += 400;
					i += 2;
				}
				else
				{
					num += 100;
					i++;
				}
			}
			else if (s[i] == 'L')
			{
				num += 50;
				i++;
			}
			else if (s[i] == 'X')
			{
				if (s[i + 1] == 'C' && i != s.size() - 1)
				{
					num += 90;
					i += 2;
				}
				else if (s[i + 1] == 'L' && i != s.size() - 1)
				{
					num += 40;
					i += 2;
				}
				else
				{
					num += 10;
					i++;
				}
			}
			else if (s[i] == 'V')
			{
				num += 5;
				i++;
			}
			else if (s[i] == 'I')
			{
				if (s[i + 1] == 'X' && i != s.size() - 1)
				{
					num += 9;
					i += 2;
				}
				else if (s[i + 1] == 'V' && i != s.size() - 1)
				{
					num += 4;
					i += 2;
				}
				else
				{
					num += 1;
					i++;
				}
			}
		}
		return num;
	}

	int romamToInt2(string s)
	{
		unordered_map<char, int> hash;
		hash['I'] = 1;
		hash['V'] = 5;
		hash['X'] = 10;
		hash['L'] = 50;
		hash['C'] = 100;
		hash['D'] = 500;
		hash['M'] = 1000;

		int num = 0;
		int i = 0;
		int cur = 0;
		int nxt = 0;
		while (i < s.size())
		{
			cur = hash[s[i]];
			if (i == s.size() - 1)
			{
				num += cur;
				break;
			}
			nxt = hash[s[i + 1]];
			if (cur < nxt)
			{
				num += (nxt - cur);
				i += 2;
			}
			else
			{
				num += cur;
				i += 1;
			}

		}
		return num;
	}
};


/*-----------------------------------------------------
	leetcode 14
------------------------------------------------------*/
class Solution14 {
public:
	string longestCommonPrefix(vector<string>& strs) {
		if (strs.size() == 0)
			return "";
		string first = strs[0];
		int len = 0;
		for (int j = 0; j < first.size(); j++)
		{
			for (int i = 0; i < strs.size(); i++)
			{
				if (strs[i].size() - 1 < j)
					return first.substr(0, len);
				if (first[j] != strs[i][j])
					return first.substr(0, len);
			}
			len++;
		}
		return first.substr(0, len);
	}

	string longestCommonPrefix2(vector<string>& strs)
	{
		if (strs.size() == 0)
			return "";
		if (strs.size() == 1)
			return strs[0];

		int low = 0;
		int high = strs[0].size() / 2;
		while (low <= high)
		{
			if (process(strs, low, high))
			{
				low = high + 1;
				high = (low + 2 * high) / 2;
			}
			else if (low == high)
				break;
			else
				high = (low + high) / 2;
		}
		return strs[0].substr(0, low);
	}

	bool process(vector<string>& strs, int low, int high)
	{
		for (int i = 1; i < strs.size(); i++)
		{
			for (int j = low; j <= high; j++)
			{
				if (strs[i].size() < j + 1)
					return false;
				if (strs[0][j] != strs[i][j])
					return false;
			}
		}
		return true;
	}

};


/*-----------------------------------------------------
	leetcode 15
------------------------------------------------------*/
class Solution15 {
public:
	vector<vector<int>> threeSum(vector<int>& nums) {
		vector<vector<int>> res;
		sort(nums.begin(), nums.end());

		int start = 0;
		int end = 0;
		int val = 0;
		for (int i = 0; i < nums.size(); i++)
		{
			if (i > 0)
				if (nums[i] == nums[i - 1])
					continue;
			start = i + 1;
			end = nums.size() - 1;
			while (start < end)
			{
				val = nums[i] + nums[start] + nums[end];
				if (val == 0)
				{
					res.push_back({ nums[i], nums[start], nums[end] });
					do
						end--;
					while (nums[end] == nums[end + 1] && end > 0);
					do
						start++;
					while (nums[start] == nums[start - 1] && start < nums.size() - 1);
				}
				else if (val > 0)
					end--;
				else
					start++;
			}
		}
		return res;

	}
};


/*-----------------------------------------------------
	leetcode 16
------------------------------------------------------*/
class Solution16 {
public:
	int threeSumClosest(vector<int>& nums, int target) {
		int best = nums[0] + nums[1] + nums[2];
		sort(nums.begin(), nums.end());

		int start = 0;
		int end = nums.size() - 1;
		int val = 0;
		for (int i = 0; i < nums.size(); i++)
		{
			start = i + 1;
			end = nums.size() - 1;
			if (i > 0)
				if (nums[i] == nums[i - 1])
					continue;
			while (start < end)
			{
				val = nums[i] + nums[start] + nums[end];
				if (val - target == 0)
					return target;
				else if (abs(val - target) < abs(best - target))
					best = val;
				if (val > target)
					end--;
				else
					start++;
			}
		}
		return best;
	}
};


/*-----------------------------------------------------
	leetcode 17
------------------------------------------------------*/
class Solution17 {
public:
	vector<string> letterCombinations(string digits) {
		vector<string> strs;
		if (digits.empty())
			return strs;
		string res = "";
		unordered_map<char, string> map{ {'2', "abc"}, {'3', "def"}, {'4', "ghi"},
										 {'5', "jkl"}, {'6', "mno"}, {'7', "pqrs"},
										 {'8', "tuv"}, {'9', "wxyz"} };
		process(digits, res, map, strs);
		return strs;
	}

	void process(string &digits, string res, unordered_map<char, string>& map, vector<string> &strs)
	{
		int start = res.size();
		string candidates = map[digits[start]];
		if (start == digits.size())
			return;

		for (char c : candidates)
		{
			if (start == digits.size() - 1)
				strs.push_back(res + c);
			process(digits, res + c, map, strs);
		}
	}
};


/*-----------------------------------------------------
	leetcode 18
------------------------------------------------------*/
class Solution18 {
public:
	vector<vector<int>> fourSum(vector<int>& nums, int target) {
		vector<vector<int>> res;

		if (nums.size() < 4)
			return res;

		sort(nums.begin(), nums.end());
		int start = 0;
		int end = 0;
		int val = 0;
		for (int i = 0; i < nums.size() - 3; i++)
		{
			if (nums[i] > target && nums[i] >= 0)
				return res;
			if (i > 0)
				if (nums[i] == nums[i - 1])
					continue;
			for (int j = i + 1; j < nums.size() - 2; j++)
			{
				if (nums[i] + nums[j] > target && nums[j] >= 0)
					break;
				if (j > i + 1)
					if (nums[j] == nums[j - 1])
						continue;
				start = j + 1;
				end = nums.size() - 1;
				while (start < end)
				{
					val = nums[i] + nums[j] + nums[start] + nums[end];
					if (val == target)
					{
						res.push_back({ nums[i], nums[j], nums[start], nums[end] });
						do
							end--;
						while (nums[end] == nums[end + 1] && end > j);
						do
							start++;
						while (nums[start] == nums[start - 1] && start < nums.size() - 1);
					}
					else if (val > target)
						end--;
					else
						start++;

				}
			}
		}
		return res;
	}
};


/*-----------------------------------------------------
	leetcode 19
------------------------------------------------------*/
struct ListNode19 {
	int val;
	ListNode19 *next;
	ListNode19(int x) : val(x), next(NULL) {}
};
class Solution19 {
public:
	ListNode19* removeNthFromEnd(ListNode19* head, int n) {
		ListNode19* Head = new ListNode19(0);
		Head->next = head;
		ListNode19* forward = Head;
		ListNode19* back = Head;
		for (int i = 0; i < n + 1; i++)
			forward = forward->next;

		while (forward)
		{
			forward = forward->next;
			back = back->next;
		}
		forward = back->next;
		back->next = back->next->next;
		if (forward)
			delete forward;
		return Head->next;
	}
};


/*-----------------------------------------------------
	leetcode 20
------------------------------------------------------*/
class Solution20 {
public:
	bool isValid(string s) {
		if (s.empty())
			return true;
		if (s.size() % 2 != 0)
			return false;
		stack<char> S;
		unordered_map<char, char> map{ {')','('}, {'}', '{'}, {']', '['} };
		for (int i = 0; i < s.size(); i++)
		{
			if (s[i] == '(' || s[i] == '{' || s[i] == '[')
				S.push(s[i]);
			else
			{
				if (S.empty())
					return false;
				if (S.top() == map[s[i]])
					S.pop();
				else
					return false;
			}
		}
		if (S.empty())
			return true;
		else
			return false;
	}
};


/*-----------------------------------------------------
	leetcode 21
------------------------------------------------------*/
struct ListNode21 {
	int val;
	ListNode21 *next;
	ListNode21(int x) : val(x), next(NULL) {}
};
class Solution21 {
public:
	ListNode21* mergeTwoLists(ListNode21* l1, ListNode21* l2) {
		ListNode21* res = new ListNode21(0);
		ListNode21* cur = res;

		while (l1 && l2)
		{
			if (l1->val <= l2->val)
			{
				cur->next = l1;
				cur = cur->next;
				l1 = l1->next;
			}
			else
			{
				cur->next = l2;
				cur = cur->next;
				l2 = l2->next;
			}
		}
		cur->next = l1 ? l1 : l2;
		return res->next;
	}
};


/*-----------------------------------------------------
	leetcode 22
------------------------------------------------------*/
class Solution22 {
public:
	vector<string> generateParenthesis(int n) {
		string res = "";
		vector<string> strs;
		process(n, 0, 0, true, res, strs);
		return strs;
	}

	void process(int& n, int l, int r, bool choose_l, string res, vector<string>& strs)
	{
		if (res.size() == 2 * n)
			strs.push_back(res);
		if (l < n)
		{
			l++;
			process(n, l, r, true, res + '(', strs);
			l--;
		}
		if (r < n && l > r)
		{
			r++;
			process(n, l, r, false, res + ')', strs);
		}
	}

};


/*-----------------------------------------------------
	leetcode 23
------------------------------------------------------*/
struct ListNode23 {
	int val;
	ListNode23 *next;
	ListNode23(int x) : val(x), next(NULL) {}
};
class Solution23 {
public:
	ListNode23* swapPairs(ListNode23* head) {
		ListNode23* pre = head;
		ListNode23* prepre = NULL;
		ListNode23* cur = NULL;
		ListNode23* nxt = NULL;
		if (!head || !head->next)
			return head;
		head = head->next;
		cur = head;
		while (cur)
		{
			if (prepre)
				prepre->next = cur;
			nxt = cur->next;
			cur->next = pre;
			pre->next = nxt;
			if (!nxt || !nxt->next)
				return head;
			prepre = pre;
			pre = nxt;
			cur = nxt->next;
		}
		return head;
	}

	ListNode23* swapPairs2(ListNode23* head)
	{
		if (!head || !head->next)
			return head;

		ListNode23* nxt = head->next;
		head->next = swapPairs2(nxt->next);
		nxt->next = head;
		return nxt;
	}
};


/*-----------------------------------------------------
	leetcode 24
------------------------------------------------------*/
class Solution24 {
public:
	int removeDuplicates(vector<int>& nums) {
		if (nums.empty())
			return 0;
		int r_idx = 1;
		for (int i = 1; i < nums.size(); i++)
		{
			if (nums[i] != nums[i - 1])
				nums[r_idx++] = nums[i];
		}
		return r_idx;
	}
};


/*-----------------------------------------------------
	leetcode 25
------------------------------------------------------*/
class Solution25 {
public:
	int removeElement(vector<int>& nums, int val) {
		if (nums.empty())
			return 0;
		int idx = 0;
		int i = 0;
		while (i < nums.size())
		{
			if (nums[i] != val)
				swap(nums[i], nums[idx++]);
			i++;
		}
		return idx;
	}
};


/*-----------------------------------------------------
	leetcode 26
------------------------------------------------------*/
class Solution26 {
public:
	int strStr(string haystack, string needle) {
		return haystack.find(needle);
	}
};


/*-----------------------------------------------------
	leetcode 27
------------------------------------------------------*/
class Solution27 {
public:
	int divide(int dividend, int divisor) {
		if (divisor == 1)
			return dividend;
		if (divisor == -1 && dividend == INT_MIN)
			return INT_MAX;
		if (divisor == -1)
			return -dividend;
		if (divisor == INT_MIN)
			return 0;
		if (dividend == INT_MIN)
			dividend = dividend + 1;//2^31/2与（2^31-1）/2结果相同

		bool isPositive = false;
		if (dividend > 0 && divisor > 0 || dividend < 0 && divisor < 0)
			isPositive = true;
		dividend = abs(dividend);
		divisor = abs(divisor);

		int cnt = 0;
		while (dividend >= divisor)
		{
			int cnt_base = 1;
			int base = divisor;
			while (dividend > base)
			{
				if (base <= INT_MAX / 2 && base < dividend / 2)
				{
					base <<= 1;
					cnt_base <<= 1;
				}
				else
					break;
			}
			dividend -= base;
			cnt += cnt_base;
		}
		if (isPositive)
			return cnt;
		else
			return -cnt;
	}
};


/*-----------------------------------------------------
leetcode 31
思路： 
1、从右往左把第一个能左移的数左移
2、重新排列右边
方法：
1、从右到左找到第一个右边大于左边的数，交换
2、把右移的数插入右边序列（因为右边序列一定是递减的）
3、反转右边序列
------------------------------------------------------*/
class Solution31 {
public:
	void nextPermutation(vector<int>& nums) {
		if (nums.size() == 0 || nums.size() == 1)
			return;
		int i = 0;
		int j = 0;
		for (i = nums.size() - 2; i >= 0; i--)
		{
			if (nums[i] < nums[i + 1])
				break;
		}
		if (i > -1)
			for (j = nums.size() - 1; j > 0; j--)
			{
				if (nums[j] > nums[i])
				{
					swap(nums[i], nums[j]);
					break;
				}
			}
		for (j = i + 1; j <= (i + nums.size() - 1) / 2; j++)
		{
			swap(nums[j], nums[nums.size() - 1 - (j - i - 1)]);
		}
	}
};



/*-----------------------------------------------------
	leetcode 33
------------------------------------------------------*/
class Solution33 {
public:
	int search(vector<int>& nums, int target) {
		return search(nums, 0, nums.size() - 1, target);
	}

	int search(vector<int>& nums, int low, int high, int target)
	{
		if (low > high)
			return -1;
		int mid = (low + high) / 2;
		if (target == nums[mid])
			return mid;
		if (nums[mid] <= nums[high])
		{
			if (target > nums[mid] && target <= nums[high])
				return search(nums, mid + 1, high, target);
			else
				return search(nums, low, mid - 1, target);
		}
		else
		{
			if (target >= nums[low] && target < nums[mid])
				return search(nums, low, mid - 1, target);
			else
				return search(nums, mid + 1, high, target);
		}
	}
};


/*-----------------------------------------------------
	leetcode 34
------------------------------------------------------*/
class Solution34 {
public:
	vector<int> searchRange(vector<int>& nums, int target) {
		vector<int> interval({ -1, -1 });
		if (nums.empty())
			return interval;
		int min = nums.size();
		int max = -1;
		search(nums, target, 0, nums.size() - 1, min, max);
		if (min <= max)
		{
			interval[0] = min;
			interval[1] = max;
		}
		return interval;
	}
	void search(vector<int>& nums, int target, int low, int high, int& min, int& max)
	{
		if (low > high)
			return;
		int mid = (low + high) / 2;
		if (target == nums[mid])
		{
			min = mid < min ? mid : min;
			max = mid > max ? mid : max;
			search(nums, target, low, mid - 1, min, max);
			search(nums, target, mid + 1, high, min, max);
		}
		else if (target < nums[mid])
			search(nums, target, low, mid - 1, min, max);
		else
			search(nums, target, mid + 1, high, min, max);
	}
};


/*-----------------------------------------------------
	leetcode 35
------------------------------------------------------*/
class Solution35 {
public:
	int searchInsert(vector<int>& nums, int target) {
		vector<int>::iterator it = nums.begin();
		while (it < nums.end())
		{
			if (*it == target)
				return distance(nums.begin(), it);
			else if (*it > target)
				break;
			it++;
		}
		it = nums.insert(it, target);
		return  distance(nums.begin(), it);
	}

	int searchInsert2(vector<int>& nums, int target)
	{
		int low = 0;
		int high = nums.size() - 1;
		int mid = (low + high) / 2;
		while (low < high)
		{
			mid = (low + high) / 2;
			if (target == nums[mid])
				return mid;
			else if (target < nums[mid])
				high = mid - 1;
			else
				low = mid + 1;
		}
		if (nums[low] >= target)
			return low;
		else
			return low + 1;
	}
};


/*-----------------------------------------------------
	leetcode 36
------------------------------------------------------*/
class Solution36 {
public:
	bool isValidSudoku(vector<vector<char>>& board) {
		unordered_set<char> R[9];//行
		unordered_set<char> C[9];//列
		unordered_set<char> B[9];//块
		char c;
		int idx;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
			{
				for (int m = 0; m < 3; m++)
					for (int n = 0; n < 3; n++)
					{
						c = board[3 * i + m][3 * j + n];
						if (c == '.')
							continue;
						if (B[3 * i + j].find(c) != B[3 * i + j].end())
							return false;
						if (R[3 * i + m].find(c) != R[3 * i + m].end())
							return false;
						if (C[3 * j + n].find(c) != C[3 * j + n].end())
							return false;
						B[3 * i + j].insert(c);
						R[3 * i + m].insert(c);
						C[3 * j + n].insert(c);
					}
			}
		return true;
	}
};


/*-----------------------------------------------------
	leetcode 38
------------------------------------------------------*/
class Solution38 {
public:
	string countAndSay(int n) {
		return process(n);
	}
	string process(int n)
	{
		if (n == 1)
			return "1";
		if (n == 2)
			return "11";

		string str = process(n - 1);
		string res = "";
		char first = '1';
		char second = str[0];
		for (int i = 1; i < str.size(); i++)
		{
			if (str[i] != str[i - 1])
			{
				res.push_back(first);
				res.push_back(second);
				first = '1';
				second = str[i];
			}
			else
			{
				first += 1;
				second = str[i];
			}
		}
		res.push_back(first);
		res.push_back(second);
		return res;
	}
};


/*-----------------------------------------------------
	leetcode 39
------------------------------------------------------*/
class Solution39 {
public:
	vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
		vector<vector<int>> res;
		vector<int> cur;
		process(candidates, target, 0, cur, res);
		return res;
	}
	void process(vector<int>& nums, int target, int start, vector<int>& cur, vector<vector<int>>& res)
	{
		if (target == 0)
			res.push_back(cur);
		else if (target < 0)
			return;

		for (int i = start; i < nums.size(); i++)
		{
			cur.push_back(nums[i]);
			process(nums, target - nums[i], i, cur, res);
			cur.pop_back();
		}
	}
};


/*-----------------------------------------------------
	leetcode 40
------------------------------------------------------*/
class Solution40 {
public:
	vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
		sort(candidates.begin(), candidates.end());
		vector<vector<int>> res;
		vector<int> cur;
		process(candidates, target, 0, cur, res);
		return res;
	}
	void process(vector<int>& nums, int target, int start, vector<int>& cur, vector<vector<int>>& res)
	{
		if (target == 0)
		{
			res.push_back(cur);
			return;
		}
		if (target < 0)
			return;

		for (int i = start; i < nums.size(); i++)
		{
			if (i > start && nums[i] == nums[i - 1])
				continue;
			cur.push_back(nums[i]);
			process(nums, target - nums[i], i + 1, cur, res);
			cur.pop_back();
		}
	}
};


/*-----------------------------------------------------
	leetcode 41
------------------------------------------------------*/
class Solution41 {
public:
	string multiply(string num1, string num2) {
		if (num1 == "0" || num2 == "0")
			return "0";

		string res = "";

		int carry = 0;
		int val = 0;
		char cur;

		int i_res = 0;
		while (i_res <= num1.size() + num2.size() - 2)
		{
			val = 0;
			for (int i = 0; i < min(num1.size(), res.size() + 1); i++)
			{
				int j = i_res - i;
				if (j > num2.size() - 1)
					continue;
				val += (num1[num1.size() - 1 - i] - '0')*(num2[num2.size() - 1 - j] - '0');
			}
			val += carry;
			carry = val / 10;
			cur = val % 10 + '0';
			res = cur + res;
			i_res++;
		}
		if (carry > 0 && carry < 10)
		{
			cur = carry + '0';
			res = cur + res;
		}
		else if (carry > 10)
		{
			cur = carry % 10 + '0';
			res = cur + res;
			cur = carry / 10 + '0';
			res = cur + res;
		}
		return res;
	}
};


/*-----------------------------------------------------
	leetcode 55
------------------------------------------------------*/
class Solution55 {
public:
	bool canJump(vector<int>& nums) {//核心思想，能到i位，一定能到i-1位，所以从后往前推，压缩目标位置
		int target = nums.size() - 1;
		for (int i = nums.size() - 2; i >= 0; i--)
		{
			if (i + nums[i] >= target)
				target = i;
		}
		return target == 0;
	}

	bool canJump2(vector<int>& nums)//核心思想：从头遍历，不断更新最大到达点
	{
		int maxreach = 0;
		if (nums.size() == 1)
			return true;
		for (int cur = 0; cur < nums.size() - 1; cur++)
		{
			if (cur > maxreach)
				break;
			if (cur + nums[cur] >= nums.size() - 1)
				return true;
			if (cur + nums[cur] > maxreach)
				maxreach = cur + nums[cur];
		}
		return false;
	}

};


/*-----------------------------------------------------
	leetcode 45
------------------------------------------------------*/
class Solution45 {
public:
	int jump(vector<int>& nums) {
		if (nums.size() == 1)
			return 1;
		return process(nums, 0);
	}
	int process(vector<int>& nums, int cur)
	{
		if (cur + nums[cur] >= nums.size() - 1)
			return 0;

		int Min = INT_MAX;
		int val = 0;
		for (int i = cur + 1; i <= cur + nums[cur]; i++)
		{
			Min = min(process(nums, i), Min);
			if (Min == 1)
				break;
		}
		return Min == INT_MAX ? Min : Min + 1;
	}
	int jump2(vector<int>& nums)
	{
		int step = 0;
		int lastpos = 0;
		int maxpos = 0;
		for (int i = 0; i < nums.size(); i++)
		{
			if (i > lastpos)
			{
				lastpos = maxpos;
				step += 1;
			}
			maxpos = max(maxpos, i + nums[i]);
		}
		return step;
	}
};


/*-----------------------------------------------------
	leetcode 46
------------------------------------------------------*/
class Solution46 {
public:
	vector<vector<int>> permute(vector<int>& nums) {
		vector<vector<int>> res;
		process(nums, 0, res);
		return res;
	}
	void process(vector<int>& nums, int cur, vector<vector<int>>& res)
	{
		if (cur == nums.size())
		{
			res.push_back(nums);
			return;
		}
		for (int i = cur; i < nums.size(); i++)
		{
			swap(nums[cur], nums[i]);
			process(nums, cur + 1, res);
			swap(nums[cur], nums[i]);
		}
	}
};


/*-----------------------------------------------------
	leetcode 47
------------------------------------------------------*/
class Solution47 {
public:
	vector<vector<int>> permuteUnique(vector<int>& nums) {
		vector<vector<int>> res;
		process(nums, 0, res);
		return res;
	}
	void process(vector<int>& nums, int cur, vector<vector<int>>& res)
	{
		if (cur == nums.size())
		{
			res.push_back(nums);
			return;
		}
		unordered_set<int> hash;
		for (int i = cur; i < nums.size(); i++)
		{
			if (hash.find(nums[i]) == hash.end())
				hash.insert(nums[i]);
			else
				continue;
			swap(nums[cur], nums[i]);
			process(nums, cur + 1, res);
			swap(nums[cur], nums[i]);
		}
	}
};


/*-----------------------------------------------------
	leetcode 48
------------------------------------------------------*/
class Solution48 {
public:
	void rotate(vector<vector<int>>& matrix) {
		int x1 = 0;
		int y1 = 0;
		int x2 = matrix.size() - 1;
		int y2 = matrix[0].size() - 1;

		while (x1 < x2)
		{
			process(matrix, x1++, y1++, x2--, y2--);
		}
	}
	void process(vector<vector<int>>& m, int x1, int y1, int x2, int y2)
	{
		int tmp = 0;
		for (int i = 0; i < x2 - x1; i++)
		{
			tmp = m[x1][y1 + i];
			m[x1][y1 + i] = m[x2 - i][y1];
			m[x2 - i][y1] = m[x2][y2 - i];
			m[x2][y2 - i] = m[x1 + i][y2];
			m[x1 + i][y2] = tmp;
		}
	}
};


/*-----------------------------------------------------
	leetcode 49
------------------------------------------------------*/
class Solution49 {
public:
	vector<vector<string>> groupAnagrams(vector<string>& strs) {
		vector<vector<string>> res;
		unordered_map<string, vector<string>*> hash;
		string key;
		for (string str : strs)
		{
			key = makeKey(str);
			if (hash.find(key) == hash.end())
			{
				hash[key] = new vector<string>;
			}
			hash[key]->push_back(str);
		}

		unordered_map<string, vector<string>*>::iterator it;
		for (it = hash.begin(); it != hash.end(); it++)
			res.push_back(*(it->second));
		return res;
	}

	string makeKey(string str)
	{
		string res = "";
		int cnt[26] = { 0 };
		for (int i = 0; i < str.size(); i++)
			cnt[str[i] - 'a']++;
		for (int i = 0; i < 26; i++)
			res += '#' + to_string(cnt[i]);

		return res;
	}
};


/*-----------------------------------------------------
	leetcode 887
------------------------------------------------------*/
class Solution887 {
public:
	int superEggDrop(int K, int N) {
		return process(K, N);
	}
	int process(int curK, int curN)
	{
		if (curK == 1 || curN == 1 || curN == 0)
			return curN;

		int curmin = curN;
		for (int i = 1; i <= curN; i++)
			curmin = min(curmin, max(process(--curK, i - 1), process(curK, curN - i)));

		return 1 + curmin;
	}

	int superEggDrop2(int K, int N)//dp
	{
		int** m = new int*[K + 1];
		for (int i = 0; i <= K; i++)
			m[i] = new int[N + 1];

		for (int n = 0; n <= N; n++)
			m[1][n] = n;
		for (int k = 1; k <= K; k++)
		{
			m[k][0] = 0;
			m[k][1] = 1;
		}
		for (int k = 2; k <= K; k++)
			for (int n = 2; n <= N; n++)
			{
				int low = 1;
				int high = n;
				int mid = (low + high) / 2;
				while (low + 1 < high)	//二分查找最小值
				{
					mid = (low + high) / 2;
					int val1 = m[k - 1][mid - 1];
					int val2 = m[k][n - mid];
					if (val1 == val2)
						break;
					else if (val1 < val2)
						low = mid;
					else
						high = mid;
				}
				m[k][n] = max(m[k - 1][mid - 1], m[k][n - mid]) + 1;
			}

		return m[K][N];
	}
};


/*-----------------------------------------------------
	leetcode 50
------------------------------------------------------*/
class Solution50 {
public:
	double myPow(double x, int n) {
		double res = 1.0;
		for (int i = n; i != 0; i /= 2) {
			if (i % 2 != 0) {
				res *= x;
			}
			x *= x;
		}
		return  n < 0 ? 1 / res : res;
	}
};


/*-----------------------------------------------------
	leetcode 51
------------------------------------------------------*/
class Solution51 {
public:
	vector<vector<string>> solveNQueens(int n) {
		vector<vector<string>> res;
		vector<string> curRes;
		vector<int> curIdx;
		string tmp;
		for (int i = 0; i < n; i++)
			tmp += '.';
		for (int i = 0; i < n; i++)
			curRes.push_back(tmp);
		process(n, res, curRes, curIdx);
		return res;
	}
	void process(int n, vector<vector<string>>& res, vector<string>& curRes, vector<int>& curIdx)
	{
		int curRow = curIdx.size();
		if (curRow == n)
		{
			res.push_back(curRes);
			curRes[curRow - 1][curIdx[curRow - 1]] = '.';
			curIdx.pop_back();
			return;
		}

		for (int curCol = 0; curCol < n; curCol++)
		{
			if (!check(n, curRes, curRow, curCol))
				continue;
			curRes[curRow][curCol] = 'Q';
			curIdx.push_back(curCol);
			process(n, res, curRes, curIdx);
		}

		/*回溯*/
		if (!curIdx.empty())
		{
			curRes[curRow - 1][curIdx[curRow - 1]] = '.';
			curIdx.pop_back();
		}
		return;
	}

	bool check(int n, vector<string>& curRes, int curRow, int curCol)
	{
		/*检查列是否可行*/
		for (int i = 0; i < curRow; i++)
			if (curRes[i][curCol] != '.')
				return false;
		/*检查左上对角线是否可行*/
		int i = curRow - 1;
		int j = curCol - 1;
		while (i >= 0 && j >= 0)
		{
			if (curRes[i][j] != '.')
				return false;
			i--;
			j--;
		}
		/*检查右上对角线是否可行*/
		i = curRow - 1;
		j = curCol + 1;
		while (i >= 0 && j <= n - 1)
		{
			if (curRes[i][j] != '.')
				return false;
			i--;
			j++;
		}
		return true;
	}
};


/*-----------------------------------------------------
	字符串分解
------------------------------------------------------*/
void StringExpansion()
{
	// e3[2[abc]fg]->eabcabcfgabcabcfgabcabcfg
	while (true)
	{
		string str;
		cin >> str;;
		stack<char> S;
		string repeat = "";
		int times = 0;
		for (int i = 0; i < str.size(); i++)
		{
			times = 0;
			repeat = "";
			if (str[i] != ']')
			{
				S.push(str[i]);
			}
			else
			{
				while (S.top() != '[')
				{
					repeat = S.top() + repeat;
					S.pop();
				}
				S.pop();
				times = S.top() - '0';
				S.pop();
				for (int j = 0; j < times; j++)
					for (int k = 0; k < repeat.size(); k++)
						S.push(repeat[k]);
			}
		}

		string res = "";
		while (!S.empty())
		{
			res = S.top() + res;
			S.pop();
		}
		cout << res << endl;
	}
}


/*-----------------------------------------------------
	leetcode 52
------------------------------------------------------*/
class Solution52 {
public:
	int totalNQueens(int n) {
		int nums = 0;
		vector<string> curRes;
		vector<int> curIdx;
		string tmp;
		for (int i = 0; i < n; i++)
			tmp += '.';
		for (int i = 0; i < n; i++)
			curRes.push_back(tmp);
		process(n, nums, curRes, curIdx);
		return nums;
	}
	void process(int n, int& nums, vector<string>& curRes, vector<int>& curIdx)
	{
		int curRow = curIdx.size();
		if (curRow == n)
		{
			nums++;;
			curRes[curRow - 1][curIdx[curRow - 1]] = '.';
			curIdx.pop_back();
			return;
		}

		for (int curCol = 0; curCol < n; curCol++)
		{
			if (!check(n, curRes, curRow, curCol))
				continue;
			curRes[curRow][curCol] = 'Q';
			curIdx.push_back(curCol);
			process(n, nums, curRes, curIdx);
		}

		/*回溯*/
		if (!curIdx.empty())
		{
			curRes[curRow - 1][curIdx[curRow - 1]] = '.';
			curIdx.pop_back();
		}
		return;
	}

	bool check(int n, vector<string>& curRes, int curRow, int curCol)
	{
		/*检查列是否可行*/
		for (int i = 0; i < curRow; i++)
			if (curRes[i][curCol] != '.')
				return false;
		/*检查左上对角线是否可行*/
		int i = curRow - 1;
		int j = curCol - 1;
		while (i >= 0 && j >= 0)
		{
			if (curRes[i][j] != '.')
				return false;
			i--;
			j--;
		}
		/*检查右上对角线是否可行*/
		i = curRow - 1;
		j = curCol + 1;
		while (i >= 0 && j <= n - 1)
		{
			if (curRes[i][j] != '.')
				return false;
			i--;
			j++;
		}
		return true;
	}
};


/*-----------------------------------------------------
	leetcode 53
------------------------------------------------------*/
class Solution53 {
public:
	int maxSubArray(vector<int>& nums) {
		int maxSum = INT_MIN;
		int sum;
		for (int i = 0; i < nums.size(); i++)
		{
			sum = process(nums, i);
			if (maxSum < sum)
				maxSum = sum;
		}
		return maxSum;
	}
	int process(vector<int>& nums, int right)
	{
		if (right == 0)
			return nums[0];
		return max(nums[right] + process(nums, right - 1), nums[right]);
	}

	int maxSubArray2(vector<int>& nums)
	{
		int maxSum = nums[0];
		for (int i = 1; i < nums.size(); i++)
		{
			nums[i] = max(nums[i], nums[i] + nums[i - 1]);
			if (maxSum < nums[i])
				maxSum = nums[i];
		}
		return maxSum;
	}

};


/*-----------------------------------------------------
	leetcode 416
------------------------------------------------------*/
class Solution416 {
public:
	bool canPartition(vector<int>& nums) {
		int sum = 0;
		for (int elem : nums)
			sum += elem;
		if (sum % 2 != 0)
			return false;
		return process(nums, 0, 0, sum / 2);
	}
	bool process(vector<int>& nums, int cur, int target, const int aim)
	{
		if (target == aim / 2)
			return true;

		if (cur == nums.size())
			return false;

		return process(nums, cur + 1, target + nums[cur], aim) || process(nums, cur + 1, target, aim);
	}

	bool canPartition2(vector<int>& nums)//dp
	{
		int sum = 0;
		for (int elem : nums)
			sum += elem;

		if (sum % 2 != 0)
			return false;

		bool **m = new bool*[nums.size() + 1];
		for (int i = 0; i <= nums.size(); i++)
			m[i] = new bool[sum + 1];

		for (int target = 0; target <= sum; target++)
			m[nums.size()][target] = (target == sum / 2);

		for (int cur = nums.size() - 1; cur >= 0; cur--)
			for (int target = 0; target <= sum; target++)
				m[cur][target] = m[cur + 1][target + nums[cur]] || m[cur + 1][target];
		return m[0][0];
	}

	bool canPartition3(vector<int>& nums)//二维改一维，更新一行时直接覆盖上一行
	{
		int sum = 0;
		for (int elem : nums)
			sum += elem;

		if (sum % 2 != 0)
			return false;

		bool *m = new bool[sum / 2 + 1];

		for (int target = 0; target <= sum; target++)
			m[target] = (target == sum / 2);

		for (int cur = nums.size() - 1; cur >= 0; cur--)
			for (int target = 0; target <= sum / 2; target++)
			{
				m[target] = m[target + nums[cur]] || m[target];
				if (m[target])
					return true;
			}
		return m[0];
	}
};


/*-----------------------------------------------------
	leetcode 322
------------------------------------------------------*/
class Solution322 {
public:
	int coinChange(vector<int>& coins, int amount) {
		int minval = process(coins, coins.size() - 1, amount);
		if (minval == INT_MAX)
			return -1;
		return minval;
	}
	int process(vector<int>& coins, int cur, int amount)
	{
		if (amount == 0)
			return 0;
		if (cur < 0 || amount < 0)
			return INT_MAX;

		int minval = process(coins, cur - 1, amount);
		if (amount - coins[cur] >= 0)
		{
			int val = process(coins, cur, amount - coins[cur]);
			if (val == INT_MAX)
				val -= 1;
			minval = min(val + 1, minval);
			val = process(coins, cur - 1, amount - coins[cur]);
			if (val == INT_MAX)
				val -= 1;
			minval = min(val + 1, minval);
		}
		return minval;
	}

	int coinChange2(vector<int>& coins, int amount)
	{
		int **m = new int*[coins.size() + 1];
		for (int i = 0; i < coins.size() + 1; i++)
			m[i] = new int[amount + 1];
		for (int target = 0; target <= amount; target++)
			m[0][target] = INT_MAX;
		for (int cur = 0; cur <= coins.size(); cur++)
			m[cur][0] = 0;

		int val;
		for (int cur = 1; cur <= coins.size(); cur++)
			for (int target = 1; target <= amount; target++)
			{
				if (cur == 3 && target == 1)
					cout << "";
				m[cur][target] = m[cur - 1][target];
				if (target - coins[cur - 1] >= 0)
				{
					val = m[cur][target - coins[cur - 1]];
					if (val == INT_MAX)
						val -= 1;
					m[cur][target] = min(m[cur][target], val + 1);
					val = m[cur - 1][target - coins[cur - 1]];
					if (val == INT_MAX)
						val -= 1;
					m[cur][target] = min(m[cur][target], val + 1);
				}
			}
		return m[coins.size()][amount] == INT_MAX ? -1 : m[coins.size()][amount];
	}
};


/*-----------------------------------------------------
	leetcode 54
------------------------------------------------------*/
class Solution54 {
public:
	vector<int> spiralOrder(vector<vector<int>>& matrix) {
		vector<int> res;
		if (matrix.empty())
			return res;
		int x1 = 0;
		int y1 = 0;
		int x2 = matrix.size() - 1;
		int y2 = matrix[0].size() - 1;
		while (x1 <= x2 && y1 <= y2)
		{
			process(matrix, res, x1, y1, x2, y2);
			x1++; y1++;
			x2--; y2--;
		}
		return res;
	}
	void process(const vector<vector<int>>& m, vector<int>& res, int x1, int y1, int x2, int y2)
	{

		if (x1 == x2 && y1 == y2)
			res.push_back(m[x1][x2]);
		else if (x1 == x2)
			for (int i = y1; i <= y2; i++)
				res.push_back(m[x1][i]);
		else if (y1 == y2)
			for (int i = x1; i <= x2; i++)
				res.push_back(m[i][y1]);
		else
		{
			int cnt = 0;
			int curx = x1;
			int cury = y1;
			while (cnt < 2 * (x2 - x1 + y2 - y1))
			{
				res.push_back(m[curx][cury]);
				cnt++;
				if (curx == x1 && cury != y2)
					cury++;
				else if (curx != x2 && cury == y2)
					curx++;
				else if (curx == x2 && cury != y1)
					cury--;
				else
					curx--;
			}
		}
	}
};


/*-----------------------------------------------------
	leetcode 56
------------------------------------------------------*/
struct Interval {
	int start;
	int end;
	Interval() : start(0), end(0) {}
	Interval(int s, int e) : start(s), end(e) {}
};
bool operator < (Interval a, Interval b)
{
	return a.start < b.start;
}
class Solution56 {
public:

	vector<Interval> merge(vector<Interval>& intervals) {
		vector<Interval> res;
		sort(intervals.begin(), intervals.end());
		for (int i = 0; i < intervals.size(); i++)
		{
			if (i == intervals.size() - 1)
			{
				res.push_back(intervals[i]);
				break;
			}
			if (intervals[i].start == intervals[i + 1].start)
				intervals[i + 1].end = max(intervals[i].end, intervals[i + 1].end);
			else if (intervals[i].end >= intervals[i + 1].start)
			{
				intervals[i + 1].start = intervals[i].start;
				intervals[i + 1].end = max(intervals[i].end, intervals[i + 1].end);
			}
			else
				res.push_back(intervals[i]);
		}
		return res;
	}
};


/*-----------------------------------------------------
	leetcode 58
------------------------------------------------------*/
class Solution58 {
public:
	int lengthOfLastWord(string s) {
		if (s.empty())
			return 0;
		int i = s.size() - 1;
		while (i >= 0 && s[i] == ' ')
			i--;
		int cnt = 0;
		while (i >= 0 && s[i] != ' ')
		{
			i--;
			cnt++;
		}
		return cnt;
	}
};


/*-----------------------------------------------------
	leetcode 59
------------------------------------------------------*/
class Solution59 {
public:
	vector<vector<int>> generateMatrix(int n) {
		vector<vector<int>> m(n, vector<int>(n));
		int x1 = 0;
		int y1 = 0;
		int x2 = n - 1;
		int y2 = n - 1;
		int last = 0;
		while (x1 <= x2)
			last = process(m, x1++, y1++, x2--, y2--, last);
		return m;
	}
	int process(vector<vector<int>>& m, int x1, int y1, int x2, int y2, int last)
	{
		if (x1 == x2)
		{
			m[x1][y1] = ++last;
			return last;
		}
		int curx = x1;
		int cury = y1;
		int cnt = 0;
		while (cnt < 2 * (x2 - x1 + y2 - y1))
		{
			m[curx][cury] = ++last;
			if (curx == x1 && cury != y2)
				cury++;
			else if (cury == y2 && curx != x2)
				curx++;
			else if (curx == x2 && cury != y1)
				cury--;
			else
				curx--;
			cnt++;
		}
		return last;
	}

};


/*-----------------------------------------------------
	leetcode 60
------------------------------------------------------*/
class Solution60 {
public:
	string getPermutation(int n, int k) {
		int factorial = 1;
		int preNum = 2;
		for (; k >= factorial; preNum++)
			factorial *= preNum;
		factorial /= (preNum - 1);
		preNum -= 2;
		int times = k - factorial;

		string res = "";
		for (int i = n; i > n - preNum; i--)
		{
			char c = '0' + i;
			res += c;
		}
		for (int i = n - preNum; i >= 1; i--)
		{
			char c = '0' + i;
			res = c + res;
		}

		for (int i = 0; i < times; i++)
			nxtBigger(res);

		return res;
	}

	void nxtBigger(string& res)
	{
		int i = res.size() - 2;
		for (; i >= 0; i--)
			if (res[i] < res[i + 1])
				break;

		int j = i + 1;
		for (int k = j + 1; k < res.size(); k++)
		{
			if (res[k] > res[i] && res[k] < res[j])
				j = k;
			if (res[k] == res[i] + 1)
				break;
		}

		swap(res[i], res[j]);
		reverse(res, i + 1);
	}

	void reverse(string& str, int start)
	{
		int end = str.size() - 1;
		while (start < end)
			swap(str[start++], str[end--]);
	}
};


/*-----------------------------------------------------
	leetcode 61
------------------------------------------------------*/
struct ListNode61 {
	int val;
	ListNode61 *next;
	ListNode61(int x) : val(x), next(NULL) {}
};
class Solution61 {
	/*
	旋转k步，则forward指针先走k-1步，back再随之一起走,
	当forward到达最后一个节点时，back到达新的首节点，
	若forward先走k步，则back指向新的尾节点。
	*/
public:
	ListNode61* rotateRight(ListNode61* head, int k) {
		if (!head)
			return NULL;
		ListNode61* forward = head;
		ListNode61* back = head;
		ListNode61* tail = NULL;
		for (int i = 0; i < k; i++)
		{
			if (!forward->next)
			{
				tail = forward;
				tail->next = head;
				k = k % (i + 1);
				i = -1;
			}
			forward = forward->next;
		}
		while (forward->next && forward != tail)
		{
			forward = forward->next;
			back = back->next;
		}
		forward->next = head;
		forward = back->next;
		back->next = NULL;
		return forward;
	}
};


/*-----------------------------------------------------
	leetcode 62
------------------------------------------------------*/
class Solution62 {
public:
	int uniquePaths(int m, int n) {
		return process(0, 0, m, n);
	}
	int process(int x, int y, int m, int n)
	{
		if (y == m - 1 && x == n - 1)
			return 1;

		if (y == m - 1)
			return process(x + 1, y, m, n);

		if (x == n - 1)
			return process(x, y + 1, m, n);

		return process(x, y + 1, m, n) + process(x + 1, y, m, n);
	}

	int uniquePaths2(int m, int n)
	{
		vector<vector<int>> dp(m, vector<int>(n));
		dp[m - 1][n - 1] = 1;
		for (int x = n - 2; x >= 0; x--)
			dp[m - 1][x] = dp[m - 1][x + 1];
		for (int y = m - 2; y >= 0; y--)
			dp[y][n - 1] = dp[y + 1][n - 1];

		for (int y = m - 2; y >= 0; y--)
			for (int x = n - 2; x >= 0; x--)
				dp[y][x] = dp[y + 1][x] + dp[y][x + 1];

		return dp[0][0];
	}
};


/*-----------------------------------------------------
	leetcode 63
------------------------------------------------------*/
class Solution63 {
public:
	int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
		return process(obstacleGrid, 0, 0);
	}
	int process(vector<vector<int>>& grid, int y, int x)
	{
		int height = grid.size();
		int width = grid[0].size();

		if (y == height - 1 && x == width - 1)
			return 1;

		if (grid[y][x] == 1)
			return 0;

		if (y == height - 1)
			return process(grid, y, x + 1);

		if (x == width - 1)
			return process(grid, y + 1, x);

		return process(grid, y, x + 1) + process(grid, y + 1, x);
	}

	long uniquePathsWithObstacles2(vector<vector<int>>& obstacleGrid)
	{
		int height = obstacleGrid.size();
		int width = obstacleGrid[0].size();
		if (obstacleGrid[height - 1][width - 1] == 0)
			return 0;
		vector<vector<long>> dp(height, vector<long>(width));

		for (int y = height - 1; y >= 0; y--)
			for (int x = width - 1; x >= 0; x--)
			{
				if (x == width - 1 && y == height - 1)
					dp[y][x] = 1;
				else if (obstacleGrid[y][x] == 1)
					dp[y][x] = 0;
				else if (y == height - 1)
					dp[y][x] = dp[y][x + 1];
				else if (x == width - 1)
					dp[y][x] = dp[y + 1][x];
				else
					dp[y][x] = dp[y + 1][x] + dp[y][x + 1];
			}

		return dp[0][0];
	}
};


/*-----------------------------------------------------
	leetcode 64
------------------------------------------------------*/
class Solution64 {
public:
	int minPathSum(vector<vector<int>>& grid) {
		int height = grid.size();
		int width = grid[0].size();
		return process(grid, 0, 0, height, width);
	}
	int process(vector<vector<int>>& grid, int y, int x, int height, int width)
	{
		if (y == height - 1 && x == width - 1)
			return grid[y][x];

		if (y == height - 1)
			return process(grid, y, x + 1, height, width) + grid[y][x];

		if (x == width - 1)
			return process(grid, y + 1, x, height, width) + grid[y][x];

		return min(process(grid, y, x + 1, height, width) + grid[y][x + 1], process(grid, y + 1, x, height, width) + grid[y + 1][x]);
	}

	int minPathSum2(vector<vector<int>>& grid)
	{
		int height = grid.size();
		int width = grid[0].size();
		vector<vector<int>> dp(height, vector<int>(width));

		dp[height - 1][width - 1] = grid[height - 1][width - 1];
		for (int y = height - 2; y >= 0; y--)
			dp[y][width - 1] = dp[y + 1][width - 1] + grid[y][width - 1];
		for (int x = width - 2; x >= 0; x--)
			dp[height - 1][x] = dp[height - 1][x + 1] + grid[height - 1][x];

		for (int y = height - 2; y >= 0; y--)
			for (int x = width - 2; x >= 0; x--)
				dp[y][x] = min(dp[y + 1][x], dp[y][x + 1]) + grid[y][x];

		return dp[0][0];
	}
};


/*----------------------------------------
		leetcode 66
 ---------------------------------------*/
class Solution66 {
public:
	vector<int> plusOne(vector<int>& digits) {
		int carry = 1;
		int val = 0;
		for (int i = digits.size() - 1; i >= 0; i--)
		{
			val = digits[i] + carry;
			digits[i] = val % 10;
			carry = val / 10;
		}
		if (carry == 1)
			digits.insert(digits.begin(), carry);
		return digits;
	}
};


/*----------------------------------------
		leetcode 67
 ---------------------------------------*/
class Solution67 {
public:
	string addBinary(string a, string b) {
		string res = "";
		int carry = 0;
		int i = 0;
		while (i < a.size() || i < b.size())
		{
			char cur = '0';
			int cura = 0;
			int curb = 0;
			if (i < a.size())
				cura = a[a.size() - 1 - i] - '0';
			if (i < b.size())
				curb = b[b.size() - 1 - i] - '0';
			cur += (cura + curb + carry) % 2;
			carry = (cura + curb + carry) / 2;
			res = cur + res;
			i++;
		}

		return carry == 0 ? res : '1' + res;
	}
};


/*----------------------------------------
		leetcode 69
 ---------------------------------------*/
class Solution69 {
public:
	int mySqrt(int x) {
		int low = 0;
		int high = x;
		int mid = (low + high) / 2;
		while (low < high)
		{
			if (mid >= 46341)
				high = mid - 1;
			else if (mid * mid <= x && (mid == 46340 || (mid + 1) * (mid + 1) > x))
				break;
			else if (mid * mid > x)
				high = mid - 1;
			else if ((mid + 1) * (mid + 1) <= x)
				low = mid + 1;
			mid = (low + high) / 2;
		}
		return mid;
	}
};


/*----------------------------------------
		leetcode 70
 ---------------------------------------*/
class Solution70 {
public:
	/*二分法，也可用牛顿迭代法*/
	int climbStairs(int n) {
		return process(n, 0);
	}
	int process(int n, int cur)
	{
		if (cur == n)
			return 1;
		else if (cur == n - 1)
			return process(n, cur + 1);
		else
			return process(n, cur + 1) + process(n, cur + 2);
	}

	int climbStairs2(int n)
	{
		vector<int> dp(n + 1);
		dp[n] = 1;
		dp[n - 1] = 1;
		for (int cur = n - 2; cur >= 0; cur--)
			dp[cur] = dp[cur + 1] + dp[cur + 2];
		return dp[0];
	}
};


/*----------------------------------------
		leetcode 71
 ---------------------------------------*/
class Solution71 {
public:
	string simplifyPath(string path) {
		stack<string> S;
		string tmp = "";
		for (int i = 0; i <= path.size(); i++)
		{
			if (path[i] == '/' || i == path.size())
			{
				if (tmp == "..")
				{
					if (!S.empty())
						S.pop();
				}
				else if (tmp != "" && tmp != ".")
					S.push(tmp);
				tmp = "";
			}
			else
				tmp += path[i];
		}


		string res;
		while (!S.empty())
		{
			res = '/' + S.top() + res;
			S.pop();
		}
		return res == "" ? "/" : res;
	}
};


/*----------------------------------------
		leetcode 73
 ---------------------------------------*/
class Solution73 {
public:
	void setZeroes(vector<vector<int>>& matrix) {
		bool firstrow = false;
		bool firstcol = false;
		/*记录第一行和第一列是否需要置0*/
		for (int i = 0; i < matrix.size(); i++)
			if (matrix[i][0] == 0)
			{
				firstcol = true;
				break;
			}
		for (int j = 0; j < matrix[0].size(); j++)
			if (matrix[0][j] == 0)
			{
				firstrow = true;
				break;
			}
		/*用第一列记录每一行是否需要置0，第一行记录每一列*/
		for (int i = 1; i < matrix.size(); i++)
			for (int j = 0; j < matrix[0].size(); j++)
				if (matrix[i][j] == 0)
				{
					matrix[i][0] = 0;
					break;
				}
		for (int j = 1; j < matrix[0].size(); j++)
			for (int i = 0; i < matrix.size(); i++)
				if (matrix[i][j] == 0)
				{
					matrix[0][j] = 0;
					break;
				}
		/*除了第一行和第一列外，置0与否*/
		for (int i = 1; i < matrix.size(); i++)
			if (matrix[i][0] == 0)
				for (int j = 1; j < matrix[0].size(); j++)
					matrix[i][j] = 0;
		for (int j = 1; j < matrix[0].size(); j++)
			if (matrix[0][j] == 0)
				for (int i = 0; i < matrix.size(); i++)
					matrix[i][j] = 0;
		/*第一行和第一列，置零与否*/
		if (firstrow)
			for (int j = 0; j < matrix[0].size(); j++)
				matrix[0][j] = 0;
		if (firstcol)
			for (int i = 0; i < matrix.size(); i++)
				matrix[i][0] = 0;
	}
};


/*----------------------------------------
		leetcode 74
 ---------------------------------------*/
class Solution74 {
public:
	bool searchMatrix(vector<vector<int>>& matrix, int target) {
		if (matrix.empty() || matrix[0].empty())
			return false;
		int i = 0;
		int j = matrix[0].size() - 1;
		while (true)
		{
			if (matrix[i][j] == target)
				return true;
			else if (matrix[i][j] > target)
			{
				if (j != 0)
					j--;
				else
					return false;
			}
			else
			{
				if (i != matrix.size() - 1)
					i++;
				else
					return false;
			}
		}
	}
};


/*----------------------------------------
		leetcode 75
 ---------------------------------------*/
class Solution75 {
public:
	void sortColors(vector<int>& nums) {//桶排法
		vector<int> n(3, 0);
		for (int i = 0; i < nums.size(); i++)
			n[nums[i]]++;
		n[1] += n[0];
		n[2] += n[1];
		for (int i = 0; i < nums.size(); i++)
		{
			if (i < n[0])
				nums[i] = 0;
			else if (i < n[1])
				nums[i] = 1;
			else
				nums[i] = 2;
		}
	}

	void sortColors2(vector<int>& nums)//荷兰国旗法
	{
		int less = -1;
		int more = nums.size();
		int cur = 0;
		while (cur < more)
		{
			if (nums[cur] == 0)
				swap(nums[cur++], nums[++less]);
			else if (nums[cur] == 1)
				cur++;
			else
				swap(nums[cur], nums[--more]);
		}

	}
};


/*----------------------------------------
		leetcode 76
 ---------------------------------------*/
class Solution76 {
public:
	string minWindow(string s, string t) {
		if (s.size() < t.size())
			return "";
		vector<int> cntt(58, 0);
		for (int i = 0; i < t.size(); i++)
			cntt[t[i] - 'A']++;
		vector<int> cnt(58, 0); //ascii:65~122
		int start = 0;
		int end = 0;
		int beststart = 0;
		int minlen = INT_MAX;
		cnt[s[0] - 'A']++;
		while (end < s.size())
		{
			int len = end - start + 1;
			if (len < t.size())
			{
				if (end < t.size())
					cnt[s[++end] - 'A']++;
				else
					break;
			}
			else if (len == t.size())
			{
				if (check(t, cnt, cntt))
					return s.substr(start, len);
				else if (end == s.size() - 1)
					end++;
				else
					cnt[s[++end] - 'A']++;
			}
			else
			{
				bool finalstep = false;
				while (check(t, cnt, cntt))
				{
					finalstep = true;
					cnt[s[start++] - 'A']--;
				}
				if (!finalstep)
					if (end == s.size() - 1)
						end++;
					else
						cnt[s[++end] - 'A']++;
				else
				{
					len = end - start + 2;
					if (len < minlen)
					{
						beststart = start - 1;
						minlen = len;
					}
					finalstep = false;
				}
			}
		}
		if (minlen == INT_MAX)
			minlen = 0;
		return s.substr(beststart, minlen);
	}

	bool check(string& t, vector<int> cnt, vector<int>& cntt)
	{
		bool is = true;
		for (int i = 0; i < 58; i++)
		{
			if (cnt[i] < cntt[i])
			{
				is = false;
				break;
			}
		}
		return is;
	}
};


/*----------------------------------------
		leetcode 77
 ---------------------------------------*/
class Solution77 {
public:
	vector<vector<int>> combine(int n, int k) {
		vector<vector<int>> res;
		vector<int> cur;
		for (int i = 1; n - i >= k - 1; i++)
			process(res, cur, i, k - 1, n);
		return res;
	}
	void process(vector<vector<int>>& res, vector<int> cur, int i, int k, const int n)
	{
		cur.push_back(i);
		if (k == 0)
		{
			res.push_back(cur);
			return;
		}
		for (int j = i + 1; n - j >= k - 1; j++)
			process(res, cur, j, k - 1, n);
	}
};


/*----------------------------------------
		leetcode 78
 ---------------------------------------*/
class Solution78 {
public:
	vector<vector<int>> subsets(vector<int>& nums) {
		vector<vector<int>> res;
		vector<int> cur;
		res.push_back({});
		int N = nums.size();
		for (int num = 1; num <= N; num++)
			for (int i = 0; N - 1 - i >= num - 1; i++)
				process(nums, res, cur, i, num - 1, nums.size());
		return res;
	}
	void process(vector<int>& nums, vector<vector<int>>& res, vector<int> cur, int i, int k, const int n)
	{
		cur.push_back(nums[i]);
		if (k == 0)
		{
			res.push_back(cur);
			return;
		}
		for (int j = i + 1; n - 1 - j >= k - 1; j++)
			process(nums, res, cur, j, k - 1, n);
	}
};


/*----------------------------------------
		leetcode 79
 ---------------------------------------*/
class Solution79 {
public:
	bool exist(vector<vector<char>>& board, string word) {
		if (board.empty() || board[0].empty())
		{
			if (word.empty())
				return true;
			else
				return false;
		}
		if (board.size() * board[0].size() < word.size())
			return false;

		vector<vector<bool>> access(board.size(), vector<bool>(board[0].size(), true));
		for (int i = 0; i < board.size(); i++)
			for (int j = 0; j < board[0].size(); j++)
			{
				if (process(board, access, word, i, j, 0))
					return true;
			}
		return false;
	}
	bool process(vector<vector<char>>& board, vector<vector<bool>> access, string& word, int i, int j, int idx)
	{
		if (i < 0 || i >= board.size() || j < 0 || j >= board[0].size())
			return false;
		if (!access[i][j] || board[i][j] != word[idx])
			return false;
		else if (idx == word.size() - 1)
			return true;

		access[i][j] = false;
		if (process(board, access, word, i - 1, j, idx + 1))
			return true;
		if (process(board, access, word, i + 1, j, idx + 1))
			return true;
		if (process(board, access, word, i, j - 1, idx + 1))
			return true;
		if (process(board, access, word, i, j + 1, idx + 1))
			return true;
		access[i][j] = true;
		return false;
	}
};


/*----------------------------------------
		leetcode 80
 ---------------------------------------*/
class Solution80 {
public:
	int removeDuplicates(vector<int>& nums) {
		if (nums.size() < 3)
			return nums.size();
		int i = 2;
		for (int cur = 2; cur < nums.size(); cur++)
		{
			if (nums[cur] != nums[i - 2])
				nums[i++] = nums[cur];
		}
		return i;
	}
};


/*----------------------------------------
		leetcode 81
 ---------------------------------------*/
class Solution81 {
public:
	/*以mid为界将数组分为两半，其中最多只有一半是无序的，
	 每次只需判断是否在有序的这一半即可。*/
	bool search(vector<int>& nums, int target) {
		int low = 0, high = nums.size() - 1;
		int mid = (low + high) / 2;
		while (low <= high)
		{
			if (nums[mid] == target)
				return true;
			int midShifted = mid;
			while (midShifted + 1 < high && nums[midShifted] == nums[high])
				midShifted++;   //防止形如[1,1,3,1]的区间出现，判断时把左边两个1去掉
			if (nums[midShifted] == target)
				return true;
			if (nums[midShifted] <= nums[high])
			{
				if (nums[midShifted] < target && nums[high] >= target)
					low = midShifted + 1;
				else
					high = midShifted - 1;
			}
			else
			{
				if (nums[low] <= target && nums[mid] > target)
					high = mid - 1;
				else
					low = mid + 1;
			}
			mid = (low + high) / 2;
		}
		return false;
	}
};


/*----------------------------------------
		leetcode 83
 ---------------------------------------*/
struct ListNode83 {
	int val;
	ListNode83* next;
	ListNode83(int x) : val(x), next(NULL) {}
};
class Solution {
public:
	ListNode83* deleteDuplicates(ListNode83* head) {
		if (!head)
			return NULL;
		ListNode83* cur = head->next;
		ListNode83* pre = head;
		ListNode83* nxt = NULL;
		while (cur)
		{
			if (cur->val == pre->val)
			{
				nxt = cur;
				pre->next = cur->next;
				delete nxt;
				cur = pre->next;
			}
			else
			{
				pre = pre->next;
				cur = cur->next;
			}
		}
		return head;
	}
};


/*----------------------------------------
		leetcode 82
 ---------------------------------------*/
struct ListNode82 {
	int val;
	ListNode82* next;
	ListNode82(int x) : val(x), next(NULL) {}
};
class Solution82 {
public:
	ListNode82* deleteDuplicates(ListNode82* head) {
		if (!head)
			return NULL;
		ListNode82* node0 = new ListNode82(0); //定义头结点，方便统一操作
		node0->next = head;
		ListNode82* left = head;
		ListNode82* right = left;
		ListNode82* pre = node0;
		while (right)
		{
			while (right && right->val == left->val)
				right = right->next;
			if (left->next != right)
			{
				pre->next = right;
				left = right;
			}
			else
			{
				pre = left;
				left = right;
			}
		}
		return node0->next;
	}
};


/*----------------------------------------
		leetcode 84
 ---------------------------------------*/
class Solution84 {
public:
	/*
	核心：对每根柱子循环，求包含该柱子的面积最大值。
	方法：使用单调栈，记录柱子i左侧所有小于它高度的位置。
	在循环到柱子i时、弹出柱子idx时，idx最大面积的宽为（S中下一个元素+1, i-1）。
	 */
	int largestRectangleArea(vector<int>& heights) {
		heights.push_back(0);
		stack<int> S;
		int maxArea = 0;
		for (int i = 0; i < heights.size(); i++)
		{
			while (!S.empty() && heights[i] <= heights[S.top()])
			{
				int k = S.top();
				S.pop();
				maxArea = max(maxArea, (S.empty() ? i : ((i - 1) - (S.top() + 1) + 1)) * heights[k]);
			}
			S.push(i);
		}
		return maxArea;
	}
};


/*----------------------------------------
		leetcode 85
 ---------------------------------------*/
class Solution85 {
public:
	int maximalRectangle(vector<vector<char>>& matrix) {
		if (matrix.empty())
			return 0;
		vector<vector<int>> m(matrix.size() + 1, vector<int>(matrix[0].size(), 0));
		for (int col = m[0].size() - 1; col >= 0; col--)
			for (int row = 0; row < m.size() - 1; row++)
			{
				if (col == m[0].size() - 1)
					m[row][col] = matrix[row][col] - '0';
				else if (matrix[row][col] == '0')
					m[row][col] = 0;
				else
					m[row][col] = m[row][col + 1] + 1;
			}
		int maxArea = 0;
		for (int col = 0; col < m[0].size(); col++)
			maxArea = max(maxArea, process(m, col));
		return maxArea;
	}

	int process(vector<vector<int>>& m, int col)
	{
		stack<int> S;
		int maxArea = 0;
		for (int i = 0; i < m.size(); i++)
		{
			while (!S.empty() && m[i][col] < m[S.top()][col])
			{
				int k = S.top();
				S.pop();
				maxArea = max(maxArea, (S.empty() ? i : ((i - 1) - (S.top() + 1) + 1)) * m[k][col]);
			}
			S.push(i);
		}
		return maxArea;
	}
};


/*----------------------------------------
		leetcode 221
 ---------------------------------------*/
class Solution221 {
public:
	int maximalSquare(vector<vector<char>>& matrix) {
		int maxLen = 0;
		for (int i = 0; i < matrix.size(); i++)
			for (int j = 0; j < matrix[0].size(); j++)
			{
				maxLen = max(maxLen, process(matrix, i, j));
			}
		return maxLen;
	}
	int process(vector<vector<char>>& m, int i, int j)
	{
		if (m[i][j] == 0)
			return 0;
		if (i == 0 || j == 0)
			return 1;
		int minLen = 0;
		minLen = min(process(m, i - 1, j), process(m, i, j - 1));
		return 1 + min(minLen, process(m, i - 1, j - 1));
	}

	int maximalSquare2(vector<vector<char>>& matrix)
	{
		if (matrix.empty())
			return 0;
		vector<vector<int>> dp(matrix.size(), vector<int>(matrix[0].size(), 0));
		int maxLen = 0;
		for (int i = 0; i < dp.size(); i++)
			for (int j = 0; j < dp[0].size(); j++)
			{
				if (matrix[i][j] == '0')
					dp[i][j] = 0;
				else if (i == 0 || j == 0)
					dp[i][j] = 1;
				else
				{
					dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]);
					dp[i][j] = min(dp[i][j], dp[i - 1][j - 1]) + 1;
				}
				maxLen = max(maxLen, dp[i][j]);
			}
		return maxLen * maxLen;
	}
};


/*----------------------------------------
		leetcode 86
 ---------------------------------------*/
struct ListNode86 {
	int val;
	ListNode86* next;
	ListNode86(int x) : val(x), next(NULL) {}
};
class Solution86 {
public:
	ListNode86* partition(ListNode86* head, int x) {
		ListNode86* list1 = new ListNode86(0);
		ListNode86* list2 = new ListNode86(0);
		ListNode86* cur1 = list1;
		ListNode86* cur2 = list2;
		ListNode86* cur = head;
		while (cur)
		{
			if (cur->val < x)
			{
				cur1->next = cur;
				cur1 = cur1->next;
			}
			else
			{
				cur2->next = cur;
				cur2 = cur2->next;
			}
			cur = cur->next;
		}
		cur2->next = NULL;
		cur1->next = list2->next;
		return list1->next;
	}
};


/*----------------------------------------
		leetcode 87
 ---------------------------------------*/
class Solution87 {
public:
	bool isScramble(string s1, string s2) {
		return process(s1, s2);
	}
	bool process(string s1, string s2)
	{
		vector<int> cnt(26, 0);
		for (char elem : s1)
			cnt[elem - 'a']++;
		for (char elem : s2)
		{
			cnt[elem - 'a']--;
			if (cnt[elem - 'a'] < 0)
				return false;
		}
		if (s1.size() == 1 || s1.size() == 2)
			return true;

		for (int i = 1; i < s1.size(); i++)
		{
			if ((process(s1.substr(0, i), s2.substr(0, i)) &&
				process(s1.substr(i, s1.size() - i), s2.substr(i, s1.size() - i))))
				return true;
			if (process(s1.substr(0, i), s2.substr(s1.size() - i, i)) &&
				process(s1.substr(i, s1.size() - i), s2.substr(0, s1.size() - i)))
				return true;
		}
		return false;
	}
};


/*----------------------------------------
		leetcode 88
 ---------------------------------------*/
class Solution88 {
public:
	void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
		int cur = m + n - 1;
		int p1 = m - 1;
		int p2 = n - 1;
		while (cur >= 0)
		{
			if (p1 < 0)
				nums1[cur--] = nums2[p2--];
			else if (p2 < 0)
				nums1[cur--] = nums1[p1--];
			else if (nums1[p1] > nums2[p2])
				nums1[cur--] = nums1[p1--];
			else
				nums1[cur--] = nums2[p2--];
		}
	}
};


/*----------------------------------------
		leetcode 89
 ---------------------------------------*/
class Solution89 {
public:
	/*
	 格雷码生成公式：G(i) = i 异或 i/2,
	 */
	vector<int> grayCode(int n) {
		vector<int> res;
		res.push_back(0);
		for (int i = 1; i < 1 << n; i++)
			res.push_back(i ^ i / 2);
		return res;
	}
};


/*----------------------------------------
		找出0-n+1中缺失的元素
 ---------------------------------------*/
int findLostNum(int argc, const char* argv[]) {
	string s;
	vector<int> nums;
	getline(cin, s);
	istringstream is(s);
	int inter;
	char ch;
	while (is >> inter)
	{
		nums.push_back(inter);
		is >> ch;
	}
	if (nums[nums.size() - 1] == nums.size() - 1)
	{
		cout << nums.size();
		return 0;
	}
	int low = 0;
	int high = nums.size() - 1;
	int mid = (low + high) / 2;
	while (low < high)
	{
		if (nums[mid] == mid)
			low = mid + 1;
		else
			high = mid;
		mid = (low + high) / 2;
	}
	cout << mid;
	return 0;
}


/*----------------------------------------
		leetcode 90
 ---------------------------------------*/
class Solution90 {
public:
	vector<vector<int>> subsetsWithDup(vector<int>& nums) {
		/*
		 无重复元素时：先将空集放入res中，对num中每一个元素elem，将elem加到res中已有的子集中形成新的子集；
		 有重复元素时：先排序，若当前elem与前一个不同，同上；若相同，则仅在上一步新增的子集上加入elem形成新子集。
		 */
		sort(nums.begin(), nums.end());
		vector<vector<int>> res;
		vector<int> curRes;
		res.push_back({});
		int lastStart = 0;
		for (int i = 0; i < nums.size(); i++)
		{
			int end = res.size() - 1;
			int start = i > 0 && nums[i] == nums[i - 1] ? lastStart : 0;
			lastStart = res.size();
			for (int j = start; j <= end; j++)
			{
				curRes = res[j];
				curRes.push_back(nums[i]);
				res.push_back(curRes);
				curRes.clear();
			}
		}
		return res;
	}
};


/*----------------------------------------
		leetcode 91
 ---------------------------------------*/
class Solution91 {
public:
	int numDecodings(string s) {
		return process(s, s.size() - 1);
	}
	int process(string& s, int cur)
	{
		if (cur == 0)
		{
			if (s[0] == '0')
				return 0;
			else
				return 1;
		}
		if (cur == 1)
		{
			if ((s[0] == '0' || s[1] == '0') && (s.substr(0, 2) > "26" || s.substr(0, 2) == "00"))
				return 0;
			else if (s[0] != '0' && s[1] != '0' && s.substr(0, 2) <= "26")
				return 2;
			else
				return 1;
		}
		int cnt = 0;
		if (s[cur] != '0')
			cnt = process(s, cur - 1);
		if (s.substr(cur - 1, 2) <= "26" && s[cur - 1] != '0')
			cnt += process(s, cur - 2);
		return cnt;
	}

	int numDecodings2(string s)
	{
		if (s.size() == 1)
		{
			if (s == "0")
				return 0;
			else
				return 1;
		}
		vector<int> dp(s.size(), 0);
		if (s[0] != '0')
			dp[0] = 1;
		if ((s[0] == '0' || s[1] == '0') && (s.substr(0, 2) > "26" || s.substr(0, 2) < "10"))
			dp[1] = 0;
		else if (s[0] != '0' && s[1] != '0' && s.substr(0, 2) <= "26")
			dp[1] = 2;
		else
			dp[1] = 1;
		for (int i = 2; i < s.size(); i++)
		{
			if (s[i] != '0')
				dp[i] += dp[i - 1];
			if (s.substr(i - 1, 2) <= "26" && s[i - 1] != '0')
				dp[i] += dp[i - 2];
		}
		return dp.back();
	}
};


/*----------------------------------------
		leetcode 92
 ---------------------------------------*/
struct ListNode92 {
	int val;
	ListNode92* next;
	ListNode92(int x) : val(x), next(NULL) {}
};
class Solution92 {
public:
	ListNode92* reverseBetween(ListNode92* head, int m, int n) {
		if (!head || !head->next)
			return head;
		ListNode92* node0 = new ListNode92(0);
		node0->next = head;
		ListNode92* forward = node0;
		ListNode92* back = node0;
		ListNode92* pre;
		ListNode92* cur;
		ListNode92* nxt;
		for (int i = 0; i < n - m; i++)
			forward = forward->next;
		for (int i = 0; i < m - 1; i++)
		{
			forward = forward->next;
			back = back->next;
		}
		pre = forward->next->next;
		cur = back->next;
		for (int i = 0; i < n - m + 1; i++)
		{
			nxt = cur->next;
			cur->next = pre;
			pre = cur;
			cur = nxt;
		}
		back->next = pre;
		return node0->next;
	}
};


/*----------------------------------------
		leetcode 93
 ---------------------------------------*/
class Solution93 {
public:
	vector<string> restoreIpAddresses(string s) {
		vector<string> res;
		string ip;
		process(s, ip, 0, 4, res);
		return res;
	}
	void process(string& s, string ip, int cur, int restByte, vector<string>& res)
	{
		if (s.size() - cur < restByte)
			return;
		if (s.size() - cur > 3 * restByte)
			return;
		if (restByte == 0)
			res.push_back(ip.substr(1, ip.size() - 1));

		process(s, ip + '.' + s.substr(cur, 1), cur + 1, restByte - 1, res);
		if (cur < s.size() - 1 && s[cur] != '0')
			process(s, ip + '.' + s.substr(cur, 2), cur + 2, restByte - 1, res);
		if (cur < s.size() - 2 && s.substr(cur, 3) >= "100" && s.substr(cur, 3) <= "255")
			process(s, ip + '.' + s.substr(cur, 3), cur + 3, restByte - 1, res);
	}
};


/*----------------------------------------
		leetcode 94
 ---------------------------------------*/
struct TreeNode94 {
	int val;
	TreeNode94* left;
	TreeNode94* right;
	TreeNode94(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution94 {
public:
	vector<int> inorderTraversal(TreeNode94* root) {
		vector<int> res;
		process(root, res);
		return res;
	}
	void process(TreeNode94* t, vector<int>& res)
	{
		if (!t)
			return;
		process(t->left, res);
		res.push_back(t->val);
		process(t->right, res);
	}

	vector<int> inorderTraversal2(TreeNode94* root)
	{
		vector<int> res;
		stack<TreeNode94*> S;
		TreeNode94* t = root;
		while (!S.empty() || t)
		{
			while (t)
			{
				S.push(t);
				t = t->left;
			}
			t = S.top();
			S.pop();
			res.push_back(t->val);
			t = t->right;
		}
		return res;
	}
};


/*----------------------------------------
		leetcode 95
 ---------------------------------------*/
struct TreeNode95 {
	int val;
	TreeNode95* left;
	TreeNode95* right;
	TreeNode95(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution95 {
public:
	vector<TreeNode95*> generateTrees(int n) {
		vector<TreeNode95*> res;
		if (n == 0)
			return res;
		res = process(0, n - 1);
		return res;
	}
	vector<TreeNode95*> process(int left, int right)
	{
		vector<TreeNode95*> res;
		if (left > right)
		{
			res.push_back(NULL);
			return res;
		}
		if (left == right)
		{
			TreeNode95* newNode = new TreeNode95(left + 1);
			res.push_back(newNode);
			return res;
		}

		for (int i = left; i <= right; i++)
		{
			vector<TreeNode95*> leftTree;
			vector<TreeNode95*> rightTree;
			leftTree = process(left, i - 1);
			rightTree = process(i + 1, right);
			for (TreeNode95* leftNode : leftTree)
				for (TreeNode95* rightNode : rightTree)
				{
					TreeNode95* newNode = new TreeNode95(i + 1);
					newNode->left = leftNode;
					newNode->right = rightNode;
					res.push_back(newNode);
				}
		}
		return res;
	}
};


/*----------------------------------------
		leetcode 96
 ---------------------------------------*/
class Solution96 {
public:
	int numTrees(int n) {
		return process(0, n - 1);
	}
	int process(int left, int right)
	{
		if (left >= right)
			return 1;

		int cnt = 0;
		for (int i = left; i <= right; i++)
			cnt += process(left, i - 1) * process(i + 1, right);
		return cnt;
	}

	int numTrees2(int n)
	{
		vector<int> dp(n + 1);
		dp[0] = 1;
		dp[1] = 1;
		for (int i = 2; i <= n; i++)
		{
			dp[i] = 0;
			for (int j = 1; j <= i; j++) //j表示当前子数根节点为j
				dp[i] += dp[j - 1] * dp[i - j];
		}
		return dp.back();
	}
};


/*----------------------------------------
		leetcode 98
 ---------------------------------------*/
struct TreeNode98 {
	int val;
	TreeNode98* left;
	TreeNode98* right;
	TreeNode98(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution98 {
public:
	bool isValidBST(TreeNode98* root) {
		long pre = LONG_MIN;
		stack<TreeNode98*> S;
		while (!S.empty() || root)
		{
			while (root)
			{
				S.push(root);
				root = root->left;
			}
			root = S.top();
			if (root->val <= pre)
				return false;
			else
				pre = root->val;
			S.pop();
			root = root->right;
		}
		return true;
	}
};


/*----------------------------------------
		leetcode 100
 ---------------------------------------*/
struct TreeNode100 {
	int val;
	TreeNode100* left;
	TreeNode100* right;
	TreeNode100(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution100 {
public:
	bool isSameTree(TreeNode100* p, TreeNode100* q) {
		if (!p && !q)
			return true;
		if (!p && q || p && !q)
			return false;
		if (p->val == q->val)
			return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
		else
			return false;
	}
};


/*----------------------------------------
		leetcode 101
 ---------------------------------------*/
struct TreeNode101 {
	int val;
	TreeNode101* left;
	TreeNode101* right;
	TreeNode101(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution101 {
public:
	bool isSymmetric(TreeNode101* root) {
		if (!root)
			return true;
		return process(root->left, root->right);
	}
	bool process(TreeNode101* left, TreeNode101* right)
	{
		if (!left && !right)
			return true;
		if (!left && right || left && !right)
			return false;
		if (left->val != right->val)
			return false;
		return process(left->left, right->right) && process(left->right, right->left);
	}
};


/*----------------------------------------
		leetcode 102
 ---------------------------------------*/
struct TreeNode102 {
	int val;
	TreeNode102* left;
	TreeNode102* right;
	TreeNode102(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution102 {
public:
	vector<vector<int>> levelOrder(TreeNode102* root) {
		vector<vector<int>> res;
		vector<int> tmp;
		if (!root)
			return res;
		int last = 1;//上一层有多少节点
		int cur = 0;//当前层有多少节点
		queue<TreeNode102*> Q;
		Q.push(root);
		while (!Q.empty())
		{
			root = Q.front();
			Q.pop();
			tmp.push_back(root->val);
			last--;
			if (root->left)
			{
				Q.push(root->left);
				cur++;
			}
			if (root->right)
			{
				Q.push(root->right);
				cur++;
			}
			if (last == 0)
			{
				last = cur;
				cur = 0;
				res.push_back(tmp);
				tmp.clear();
			}
		}
		return res;
	}
};


/*----------------------------------------
		leetcode 103
 ---------------------------------------*/
struct TreeNode103 {
	int val;
	TreeNode103* left;
	TreeNode103* right;
	TreeNode103(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution103 {
public:
	vector<vector<int>> zigzagLevelOrder(TreeNode103* root) {
		vector<vector<int>> res;
		vector<int> tmp;
		if (!root)
			return res;
		int last = 1;//上一层有多少节点
		int cur = 0;//当前层有多少节点
		queue<TreeNode103*> Q;
		Q.push(root);
		while (!Q.empty())
		{
			root = Q.front();
			Q.pop();
			tmp.push_back(root->val);
			last--;
			if (root->left)
			{
				Q.push(root->left);
				cur++;
			}
			if (root->right)
			{
				Q.push(root->right);
				cur++;
			}
			if (last == 0)
			{
				last = cur;
				cur = 0;
				if (res.size() % 2 == 0)
					res.push_back(tmp);
				else
				{
					reverse(tmp.begin(), tmp.end());
					res.push_back(tmp);
				}
				tmp.clear();
			}
		}
		return res;
	}
};


/*----------------------------------------
		leetcode 104
 ---------------------------------------*/
struct TreeNode104 {
	int val;
	TreeNode104* left;
	TreeNode104* right;
	TreeNode104(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution104 {
public:
	int maxDepth(TreeNode104* root) {
		if (!root)
			return 0;
		return 1 + max(maxDepth(root->left), maxDepth(root->right));
	}
};


/*----------------------------------------
		leetcode 105
 ---------------------------------------*/
struct TreeNode105 {
	int val;
	TreeNode105* left;
	TreeNode105* right;
	TreeNode105(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution105 {
public:
	/*
	 preorder的第一个元素是当前子树的root，在inorder中找到它，左边的为左子树，右边的为右子树，递归。
	 */
	TreeNode105* buildTree(vector<int>& preorder, vector<int>& inorder) {
		return process(preorder, inorder, 0, 0, preorder.size() - 1);
	}
	TreeNode105* process(vector<int>& preorder, vector<int>& inorder, int preLeft, int inLeft, int inRight)
	{
		if (inLeft > inRight)
			return NULL;

		TreeNode105* root = new TreeNode105(preorder[preLeft]);
		for (int i = inLeft; i <= inRight; i++)
		{
			if (inorder[i] != preorder[preLeft])
				continue;
			root->left = process(preorder, inorder, preLeft + 1, inLeft, i - 1);
			root->right = process(preorder, inorder, i + 1 + (preLeft - inLeft), i + 1, inRight);
			break;
		}
		return root;
	}
};


/*----------------------------------------
		leetcode 106
 ---------------------------------------*/
struct TreeNode106 {
	int val;
	TreeNode106* left;
	TreeNode106* right;
	TreeNode106(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution106 {
public:
	TreeNode106* buildTree(vector<int>& inorder, vector<int>& postorder) {
		return process(inorder, postorder, postorder.size() - 1, 0, inorder.size() - 1);
	}
	TreeNode106* process(vector<int>& inorder, vector<int>& postorder, int postRight, int inLeft, int inRight)
	{
		if (inLeft > inRight)
			return NULL;
		TreeNode106* root = new TreeNode106(postorder[postRight]);
		for (int i = inLeft; i <= inRight; i++)
		{
			if (inorder[i] != postorder[postRight])
				continue;
			root->left = process(inorder, postorder, postRight - 1 - (inRight - i), inLeft, i - 1);
			root->right = process(inorder, postorder, postRight - 1, i + 1, inRight);
			break;
		}
		return root;
	}
};


/*----------------------------------------
		leetcode 107
 ---------------------------------------*/
struct TreeNode107 {
	int val;
	TreeNode107* left;
	TreeNode107* right;
	TreeNode107(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution107 {
public:
	vector<vector<int>> levelOrderBottom(TreeNode107* root) {
		int last = 1;
		int cur = 0;
		vector<int> layer;
		vector<vector<int>> layers;
		queue<TreeNode107*> Q;
		if (!root)
			return layers;
		Q.push(root);
		while (!Q.empty())
		{
			root = Q.front();
			layer.push_back(root->val);
			Q.pop();
			last--;
			if (root->left)
			{
				Q.push(root->left);
				cur++;
			}
			if (root->right)
			{
				Q.push(root->right);
				cur++;
			}
			if (last == 0)
			{
				last = cur;
				cur = 0;
				layers.push_back(layer);
				layer.clear();
			}
		}
		reverse(layers.begin(), layers.end());
		return layers;
	}
};


/*----------------------------------------
		leetcode 108
 ---------------------------------------*/
struct TreeNode108 {
	int val;
	TreeNode108* left;
	TreeNode108* right;
	TreeNode108(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution108 {
public:
	/*每次取数组中间元素为根，左边为左子树，右边为右子树，递归构建。*/
	TreeNode108* sortedArrayToBST(vector<int>& nums) {
		return process(nums, 0, nums.size() - 1);
	}
	TreeNode108* process(vector<int>& nums, int start, int end)
	{
		if (start > end)
			return NULL;
		int mid = (start + end) / 2;
		TreeNode108* root = new TreeNode108(nums[mid]);
		root->left = process(nums, start, mid - 1);
		root->right = process(nums, mid + 1, end);
		return root;
	}
};


/*----------------------------------------
		leetcode 109
 ---------------------------------------*/
struct ListNode109 {
	int val;
	ListNode109* next;
	ListNode109(int x) : val(x), next(NULL) {}
};
struct TreeNode109 {
	int val;
	TreeNode109* left;
	TreeNode109* right;
	TreeNode109(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution109 {
public:
	/*
	 思路同上一题，区别是list需要每次递归时用快慢指针定位根节点。
	 本题为了防止list长度为2时出现错误，直接对长度为2的情况剪纸处理，
	 实际上对于偶数长度的list，根节点取两中间位置的后一个即可。
	 */
	TreeNode109* sortedListToBST(ListNode109* head) {
		return process(head);
	}
	TreeNode109* process(ListNode109*& head)
	{
		if (!head)
			return NULL;
		if (!head->next)
			return new TreeNode109(head->val);
		if (!head->next->next)
		{
			TreeNode109* root = new TreeNode109(head->val);
			if (head->val > head->next->val)
				root->left = new TreeNode109(head->next->val);
			else
				root->right = new TreeNode109(head->next->val);
			return root;
		}

		ListNode109* back = head;
		ListNode109* forward = head;
		ListNode109* leftTail = head;
		while (forward->next && forward->next->next)
		{
			forward = forward->next->next;
			leftTail = back;
			back = back->next;
		}
		TreeNode109* root = new TreeNode109(back->val);
		leftTail->next = NULL;
		root->left = process(head);
		root->right = process(back->next);
		return root;
	}

};


/*----------------------------------------
		leetcode 110
 ---------------------------------------*/
struct TreeNode110 {
	int val;
	TreeNode110* left;
	TreeNode110* right;
	TreeNode110(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution110 {
public:
	/*递归求当前根节点的左右子树深度，若深度大于1，直接返回false*/
	bool is = true;
	bool isBalanced(TreeNode110* root) {
		process(root);
		return is;
	}
	int process(TreeNode110* root)
	{
		if (!is)
			return 0;
		if (!root)
			return 0;

		int lDeepth = 1 + process(root->left);
		int rDeepth = 1 + process(root->right);
		if (abs(lDeepth - rDeepth) <= 1)
			return max(lDeepth, rDeepth);
		else
			is = false;
		return 0;
	}
};


/*----------------------------------------
		leetcode 111
 ---------------------------------------*/
struct TreeNode111 {
	int val;
	TreeNode111* left;
	TreeNode111* right;
	TreeNode111(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution111 {
public:
	/*
	 递归计算每一个分支的深度。
	 在”递“的时候计算当前长度，目的是为了剪枝，已经不可能最短的路径不再深入。
	 */
	int minDepth(TreeNode111* root) {
		int curMinDepth = INT_MAX;
		if (!root)
			return 0;
		process(root, 1, curMinDepth);
		return curMinDepth;
	}
	void process(TreeNode111* root, int curDepth, int& curMinDepth)
	{
		if (!root->left && !root->right)
			curMinDepth = min(curMinDepth, curDepth);
		if (curDepth >= curMinDepth)
			return;
		if (root->left)
			process(root->left, curDepth + 1, curMinDepth);
		if (root->right)
			process(root->right, curDepth + 1, curMinDepth);
	}
};


/*----------------------------------------
		leetcode 112
 ---------------------------------------*/
struct TreeNode112 {
	int val;
	TreeNode112* left;
	TreeNode112* right;
	TreeNode112(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution112 {
public:
	/*
	 递归。
	 因为有负值节点，因此不能减枝。
	 */
	bool hasPathSum(TreeNode112* root, int sum) {
		if (!root)
			return false;
		if (!root->left && !root->right)
			return sum == root->val;
		return hasPathSum(root->left, sum - root->val) || hasPathSum(root->right, sum - root->val);
	}
};


/*----------------------------------------
		leetcode 113
 ---------------------------------------*/
struct TreeNode113 {
	int val;
	TreeNode113* left;
	TreeNode113* right;
	TreeNode113(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution113 {
public:
	vector<vector<int>> pathSum(TreeNode113* root, int sum) {
		vector<vector<int>> res;
		vector<int> cur;
		process(root, sum, cur, res);
		return res;

	}
	void process(TreeNode113* root, int sum, vector<int>& cur, vector<vector<int>>& res)
	{
		if (!root)
			return;

		if (!root->left && !root->right)
		{
			if (sum == root->val)
			{
				cur.push_back(root->val);
				res.push_back(cur);
				cur.pop_back();
			}
			return;
		}

		cur.push_back(root->val);
		process(root->left, sum - root->val, cur, res);
		process(root->right, sum - root->val, cur, res);
		cur.pop_back();
	}
};


/*----------------------------------------
		leetcode 114
 ---------------------------------------*/
struct TreeNode114 {
	int val;
	TreeNode114* left;
	TreeNode114* right;
	TreeNode114(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution114 {
public:
	void flatten(TreeNode114* root) {
		process(root);
	}
	void process(TreeNode114* root)
	{
		if (!root)
			return;
		process(root->left);
		process(root->right);
		TreeNode114* pre = root->left;
		if (pre)
		{
			while (pre->right)
				pre = pre->right;
			pre->right = root->right;
			root->right = root->left;
			root->left = NULL;
		}
	}
};


/*----------------------------------------
		leetcode 116
 ---------------------------------------*/
class Node116 {
public:
	int val;
	Node116* left;
	Node116* right;
	Node116* next;

	Node116() {}

	Node116(int _val, Node116* _left, Node116* _right, Node116* _next) {
		val = _val;
		left = _left;
		right = _right;
		next = _next;
	}
};
class Solution116 {
public:
	Node116* connect(Node116* root) {
		if (!root)
			return root;
		Node116* left = root->left;
		Node116* right = root->right;
		while (left)
		{
			left->next = right;
			left = left->right;
			right = right->left;
		}
		connect(root->left);
		connect(root->right);
		return root;
	}
};


/*----------------------------------------
		leetcode 117
 ---------------------------------------*/
class Node117 {
public:
	int val;
	Node117* left;
	Node117* right;
	Node117* next;

	Node117() {}

	Node117(int _val, Node117* _left, Node117* _right, Node117* _next) {
		val = _val;
		left = _left;
		right = _right;
		next = _next;
	}
};
class Solution117 {
public:
	Node117* connect(Node117* root) {
		Node117* head = new Node117(); //总是指向下一层的节点，设置头节点可以避免手动寻找每一层的第一个节点
		head->next = NULL;
		Node117* cur = root;
		Node117* tail = head;   //下一层尾
		while (cur)
		{
			while (cur)
			{
				if (cur->left)
				{
					tail->next = cur->left;
					tail = tail->next;
				}
				if (cur->right)
				{
					tail->next = cur->right;
					tail = tail->next;
				}
				cur = cur->next;
			}
			cur = head->next;
			head->next = NULL;
			tail = head;
		}
		return root;
	}
};


/*----------------------------------------
		leetcode 118
 ---------------------------------------*/
class Solution118 {
public:
	vector<vector<int>> generate(int numRows) {
		vector<vector<int>> res;
		for (int i = 0; i < numRows; i++)
		{
			vector<int> tmp(i + 1, 1);
			int num = i - 1;
			for (int j = 1; j <= num; j++)
				tmp[j] = res[i - 1][j - 1] + res[i - 1][j];
			res.push_back(tmp);
		}
		return res;
	}
};


/*----------------------------------------
		leetcode 119
 ---------------------------------------*/
class Solution119 {
public:
	vector<int> getRow(int rowIndex) {
		vector<int> res(rowIndex + 1, 1);
		for (int i = 0; i <= rowIndex; i++) //第i行
		{
			int num = i - 1; //填充num个数
			for (int j = num; j > 0; j--) //第j列
				res[j] = res[j] + res[j - 1];
		}
		return res;
	}
};


/*----------------------------------------
		leetcode 120
 ---------------------------------------*/
class Solution120 {
public:
	int minimumTotal2(vector<vector<int>>& triangle) {
		int minSum = INT_MAX;
		for (int i = 0; i < triangle.back().size(); i++)
		{
			int sum = process(triangle, triangle.size() - 1, i);
			minSum = minSum > sum ? sum : minSum;
		}
		return minSum;
	}
	int process(vector<vector<int>>& triangle, int curLayer, int curPos)
	{
		if (curLayer == 0)
			return triangle[curLayer][curPos];

		if (curPos == 0)
			return process(triangle, curLayer - 1, curPos) + triangle[curLayer][curPos];
		else if (curPos == triangle[curLayer].size() - 1)
			return process(triangle, curLayer - 1, curPos - 1) + triangle[curLayer][curPos];
		else
			return min(process(triangle, curLayer - 1, curPos - 1), process(triangle, curLayer - 1, curPos))
			+ triangle[curLayer][curPos];
	}

	int minimumTotal(vector<vector<int>>& triangle) {
		for (int curLayer = 1; curLayer < triangle.size(); curLayer++)
			for (int curPos = 0; curPos < triangle[curLayer].size(); curPos++)
			{
				if (curPos == 0)
					triangle[curLayer][curPos] += triangle[curLayer - 1][curPos];
				else if (curPos == triangle[curLayer].size() - 1)
					triangle[curLayer][curPos] += triangle[curLayer - 1][curPos - 1];
				else
					triangle[curLayer][curPos] += min(triangle[curLayer - 1][curPos], triangle[curLayer - 1][curPos - 1]);
			}
		int minSum = INT_MAX;
		for (int i = 0; i < triangle.back().size(); i++)
		{
			int curMin = triangle[triangle.size() - 1][i];
			minSum = minSum > curMin ? curMin : minSum;
		}
		return minSum;
	}
};


/*----------------------------------------
		leetcode 121
 ---------------------------------------*/
class Solution121 {
public:
	int maxProfit(vector<int>& prices) {  //复杂度N^2的算法
		int maxP = INT_MIN;
		for (int i = 0; i < prices.size(); i++)
			for (int j = i + 1; j < prices.size(); j++)
			{
				int curP = prices[j] - prices[i];
				maxP = maxP < curP ? curP : maxP;
			}
		return maxP > 0 ? maxP : 0;
	}
	int maxProfit_1(vector<int>& prices) { //空间换时间的算法
		vector<int> minPrices(prices);
		vector<int> maxPrices(prices);
		for (int i = 1; i < prices.size(); i++)
			minPrices[i] = min(minPrices[i - 1], minPrices[i]);
		for (int i = prices.size() - 2; i >= 0; i--)
			maxPrices[i] = max(maxPrices[i + 1], maxPrices[i]);

		int maxP = 0;
		for (int i = 0; i < prices.size(); i++)
		{
			int curP = maxPrices[i] - minPrices[i];
			maxP = curP > maxP ? curP : maxP;
		}
		return maxP;
	}
	int maxProfit_2(vector<int>& prices) { //递归
		vector<int> minPrices(prices);
		for (int i = 1; i < prices.size(); i++)
			minPrices[i] = min(minPrices[i - 1], minPrices[i]);
		return process(prices, minPrices, prices.size() - 1);
	}
	int process(vector<int>& prices, vector<int>& minPrices, int cur)
	{
		if (cur <= 0)
			return 0;
		return max(process(prices, minPrices, cur - 1), prices[cur] - minPrices[cur]);
	}
	int maxProfit_3(vector<int>& prices) {//DP
		vector<int> minPrices(prices);
		for (int i = 1; i < prices.size(); i++)
			minPrices[i] = min(minPrices[i - 1], minPrices[i]);

		int maxP = 0;
		for (int i = 0; i < prices.size(); i++)
			maxP = max(maxP, prices[i] - minPrices[i]);
		return maxP;
	}
	int maxProfit_4(vector<int>& prices) {//双指针
		int left = 0;
		int right = left + 1;
		int maxP = 0;
		while (left < right && right < prices.size())
		{
			if (prices[left] < prices[right])
			{
				int curP = prices[right] - prices[left];
				maxP = curP > maxP ? curP : maxP;
				right++;
			}
			else
			{
				left = right;
				right = left + 1;
			}
		}
		return maxP;
	}
};


/*----------------------------------------
		leetcode 122
 ---------------------------------------*/
class Solution122 {
public:
	int maxProfit(vector<int>& prices) {
		int left = 0;
		int right = left + 1;
		int maxP = 0;
		while (right < prices.size())
		{
			if (prices[right - 1] < prices[right])
				maxP += prices[right] - prices[right - 1];
			else
				left = right;
			right++;
		}
		return maxP;
	}
};


/*----------------------------------------
		leetcode 123
 ---------------------------------------*/
class Solution123 {
public:
	enum State { HOLD, WATCH };
	int maxProfit(vector<int>& prices) {
		return process(prices, 0, 2, WATCH);
	}
	//递归
	int process(vector<int>& prices, int curDay, int restTimes, State state)
	{
		//@curDay 当前是第几天
		//@restTimes 尚可卖出多少次
		//@state 当天状态
		if (restTimes == 0)
			return 0;
		if (curDay == prices.size())
			return 0;

		if (state == HOLD)
			return max(process(prices, curDay + 1, restTimes, HOLD),
				process(prices, curDay + 1, restTimes - 1, WATCH))
			+ prices[curDay] - prices[curDay - 1];
		return max(process(prices, curDay + 1, restTimes, HOLD),
			process(prices, curDay + 1, restTimes, WATCH));
	}
	//DP
	int maxProfit_1(vector<int>& prices) {
		vector<vector<vector<int>>> dp(prices.size() + 1, vector<vector<int>>(3, vector<int>(2, 0)));
		for (int curDay = prices.size() - 1; curDay >= 0; curDay--)
			for (int restTimes = 1; restTimes <= 2; restTimes++)
				for (int state = 0; state <= 1; state++)
				{
					if (curDay == 0 && state == 0)
						continue;
					dp[curDay][restTimes][state] = state == 0 ? max(dp[curDay + 1][restTimes][0], dp[curDay + 1][restTimes - 1][1]) + prices[curDay] - prices[curDay - 1] : max(dp[curDay + 1][restTimes][0], dp[curDay + 1][restTimes][1]);
				}
		return dp[0][2][1];
	}
};


/*----------------------------------------
		leetcode 125
 ---------------------------------------*/
class Solution125 {
public:
	void change(string& s, int i)
	{
		if (s[i] >= 'a' && s[i] <= 'z')
			s[i] = s[i] - ('z' - 'Z');
	}
	bool isLetterOrNum(string& s, int i)
	{
		return !(s[i] >= 'A' && s[i] <= 'Z' || s[i] >= 'a' && s[i] <= 'z' || s[i] >= '0' && s[i] <= '9');
	}
	bool isPalindrome(string s) {
		int left = 0;
		int right = s.size() - 1;
		while (left <= right)
		{
			while (left <= right && isLetterOrNum(s, left))
				left++;
			while (right >= left && isLetterOrNum(s, right))
				right--;
			if (left >= right)
				return true;
			change(s, left);
			change(s, right);
			if (s[left] != s[right])
				return false;
			left++;
			right--;
		}
		return true;
	}
};


/*----------------------------------------
		leetcode 127
 ---------------------------------------*/
class Solution127 {
public:
	/*暴力递归，会超时*/
	int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
		unordered_map<string, bool>* hm = new unordered_map<string, bool>();
		for (auto elem : wordList)
		{
			hm->insert(pair<string, bool>(elem, false));
		}

		(*hm)[beginWord] = true;
		int res = process(beginWord, endWord, hm);
		return res == INT_MAX ? 0 : res;
	}

	bool canTransform(string left, string right)
	{
		int cnt = 0;
		for (int i = 0; i < min(left.size(), right.size()); ++i)
		{
			if (left[i] != right[i])
				cnt++;
		}
		return cnt == 1;
	}

	int process(string cur, string& tar, unordered_map<string, bool>* hm)
	{
		if (cur == tar)
			return 1;

		int minStep = INT_MAX;
		for (unordered_map<string, bool>::iterator iter = hm->begin(); iter != hm->end(); ++iter)
		{
			if (iter->second)
				continue;
			if (canTransform(cur, iter->first))
			{
				iter->second = true;
				int forward = process(iter->first, tar, hm);
				minStep = min(minStep, forward == INT_MAX ? INT_MAX : forward + 1);
				iter->second = false;
			}
		}
		return minStep;
	}

	void createMap(vector<string>& wordList, string beginWord)
	{
		bool concludeBeginWord = false;
		for (string elem : wordList)
			if (elem == beginWord)
			{
				concludeBeginWord = true;
				break;
			}
		if (!concludeBeginWord)
			wordList.push_back(beginWord);
		for (int i = 0; i < wordList.size(); i++)
		{
			vector<int> line = vector<int>(wordList.size(), INT_MAX);
			_g.push_back(line);
			_map[wordList[i]] = i;
		}

		for (int i = 0; i < wordList.size(); i++)
		{
			for (int j = 0; j < wordList.size(); j++)
			{
				if (i == j)
					_g[i][j] = 0;
				else if (canTransform(wordList[i], wordList[j]))
				{
					_g[i][j] = 1;
					_g[j][i] = 1;
				}
			}
		}

	}

	/*图论最短路径方法，用的dijkstra，由于是无权图，其实用普通BFS就行*/
	int ladderLength2(string beginWord, string endWord, vector<string>& wordList)
	{
		/*构建图*/
		createMap(wordList, beginWord);
		if (_map.find(endWord) == _map.end())
			return 0;
		int startI = _map[beginWord];
		int endI = _map[endWord];
		vector<bool> isMin = vector<bool>(wordList.size(), false);
		isMin[startI] = true;
		for (int i = 0; i < wordList.size() - 1; i++)
		{
			int minD = INT_MAX;
			int minI = -1;
			for (int j = 0; j < wordList.size(); j++)
			{
				if (isMin[j])
					continue;
				if (_g[startI][j] <= minD)
				{
					minD = _g[startI][j];
					minI = j;
				}
			}
			isMin[minI] = true;
			if (minI == endI)
				return _g[startI][endI] == INT_MAX ? 0 : _g[startI][endI] + 1;
			else
			{
				if (minD == INT_MAX)
					return 0;
				for (int j = 0; j < wordList.size(); ++j)
				{
					if (isMin[j])
						continue;
					_g[startI][j] = min(_g[startI][j], _g[minI][j] == INT_MAX ? INT_MAX : minD + _g[minI][j]);
				}
			}
		}

		return -1;
	}


	void addAuxPoint(string word, unordered_map<string, list<string>>& auxMap)
	{
		for (int i = 0; i < word.size(); ++i)
		{
			string aux = word.substr(0, i) + '*' + word.substr(i + 1);
			if (auxMap.find(aux) == auxMap.end())
				auxMap[aux] = list<string>();
			auxMap[aux].push_back(word);
		}
		
	}

	void addLink(string word, unordered_map<string, unordered_set<string>>& map, unordered_map<string, list<string>>& auxMap)
	{
		if (map.find(word) != map.end())
			return;

		for (int i = 0; i < word.size(); ++i)
		{
			string aux = word.substr(0, i) + '*' + word.substr(i + 1);
			for (string s : auxMap[aux])
				if (s!=word)
					map[word].insert(s);
		}
	}

	/*加入辅助结点，大大减少构建图的时间*/
	int ladderLength3(string beginWord, string endWord, vector<string>& wordList)
	{
		bool flag = false;
		for (string s : wordList)
			if (s == endWord)
				flag = true;
		if (!flag)
			return 0;

		//create map with aux point
		unordered_map<string, list<string>> auxMap;
		addAuxPoint(beginWord, auxMap);
		addAuxPoint(endWord, auxMap);
		for (string elem : wordList)
			addAuxPoint(elem, auxMap);

		unordered_map<string, unordered_set<string>> map;
		addLink(beginWord, map, auxMap);
		addLink(endWord, map, auxMap);
		for (string elem : wordList)
			addLink(elem, map, auxMap);

		//BFS
		unordered_set<string> visited;
		visited.insert(beginWord);
		queue<string> q;
		q.push(beginWord);
		int distance = 1;
		int curCnt = 1;
		int nxtCnt = 0;
		while (!q.empty())
		{
			string cur = q.front();
			q.pop();
			curCnt--;
			cout << cur << " : " << distance << endl;
			for (string s : map[cur])
			{
				if (s == endWord)
					return distance+1;
				if (visited.find(s) != visited.end())
					continue;
				q.push(s);
				visited.insert(s);
				nxtCnt++;
			}
			if (curCnt == 0)
			{
				curCnt = nxtCnt;
				nxtCnt = 0;
				distance++;
			}
		}
		return 0;

	}


private:
	vector<vector<int>> _g;
	unordered_map<string, int> _map;

};


/*----------------------------------------
		leetcode 129
 ---------------------------------------*/
struct TreeNode
{
	int val;
	TreeNode* left;
	TreeNode* right;
	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
class Solution129 {
public:
	int sumNumbers(TreeNode* root) {
		if (root == NULL)
			return 0;
		int sum = 0;
		process(root, sum, 0);
		return sum;
	}
	void process(TreeNode* cur, int& sum, int curVal)
	{
		if (cur == NULL)
			return;
		else
			curVal += cur->val;
		if (cur->left == NULL && cur->right == NULL)
		{
			sum += curVal;
			return;
		}
		process(cur->left, sum, 10 * curVal);
		process(cur->right, sum, 10 * curVal);
	}
};


/*----------------------------------------
		leetcode 130
 ---------------------------------------*/
class Solution130 {
public:
	void solve(vector<vector<char>>& board) {
		unordered_set<char*> checkedList;
		for (int i = 0; i<board.size(); i++)
			for (int j = 0; j < board[0].size(); j++)
			{
				bool isSurrounded = true;
				if (board[i][j] == 'O' && checkedList.find(&board[i][j])==checkedList.end())
				{
					unordered_set<char*> curGroup;
					queue<pair<int, int>> q;
					q.push(pair<int, int>(i, j));
					while (!q.empty())
					{
						pair<int, int> curIdx = q.front();
						q.pop();
						curGroup.insert(&board[i][j]);
						checkedList.insert(&board[i][j]);
						if (isSurrounded && checkEdge(board.size(), board[0].size(), pair<int, int>(curIdx.first, curIdx.second)))
							isSurrounded = false;
						
						if (curIdx.first > 0 && board[curIdx.first - 1][curIdx.second] == 'O' && curGroup.find(&board[curIdx.first - 1][curIdx.second]) == curGroup.end())
						{
							curGroup.insert(&board[curIdx.first - 1][curIdx.second]);
							q.push(pair<int, int>(curIdx.first - 1, curIdx.second));
						}
						if (curIdx.first < board.size()-1 && board[curIdx.first + 1][curIdx.second] == 'O' && curGroup.find(&board[curIdx.first + 1][curIdx.second]) == curGroup.end())
						{
							curGroup.insert(&board[curIdx.first + 1][curIdx.second]);
							q.push(pair<int, int>(curIdx.first + 1, curIdx.second));
						}
						if (curIdx.second > 0 && board[curIdx.first][curIdx.second-1] == 'O' && curGroup.find(&board[curIdx.first][curIdx.second-1]) == curGroup.end())
						{
							curGroup.insert(&board[curIdx.first][curIdx.second-1]);
							q.push(pair<int, int>(curIdx.first, curIdx.second-1));
						}
						if (curIdx.second <board[0].size()-1 && board[curIdx.first][curIdx.second + 1] == 'O' && curGroup.find(&board[curIdx.first][curIdx.second + 1]) == curGroup.end())
						{
							curGroup.insert(&board[curIdx.first][curIdx.second + 1]);
							q.push(pair<int, int>(curIdx.first, curIdx.second + 1));
						}
					}

					if (isSurrounded)
					{
						for (auto iter = curGroup.begin(); iter != curGroup.end(); iter++)
							**iter = 'X';
					}
					curGroup.clear();
					
				}
			}

		
		
	}

private:
	bool checkEdge(int height, int width, pair<int, int> cur)
	{
		return cur.first == 0 || cur.first == height - 1 || cur.second == 0 || cur.second == width - 1;
	}
};


/*----------------------------------------
		leetcode 131
 ---------------------------------------*/
class Solution131 {
public:
	vector<vector<string>> partition(string s) {
		/*找出所有回文子串*/
		vector<vector<int>> palindrome = vector<vector<int>>(s.size(), vector<int>());//记录所有的回文串，palindrome[i]表示所有以s[i]开头的串的长度集合
		for (int i = 0; i < s.size(); ++i)//检查以s[i]为中心的奇数长度串
		{
			palindrome[i].push_back(1);
			int expand = 1;
			while (i + expand <= s.size() - 1 && i - expand >= 0 && s[i - expand] == s[i + expand])
			{
				palindrome[i - expand].push_back(2 * expand + 1);
				expand++;
			}
		}
		if (s.size()>1)
			for (int i = 0; i < s.size() - 1; ++i)//检查以s[i]和s[i+1]为中心的奇数长度串
			{
				if (s[i] == s[i + 1])
				{
					palindrome[i].push_back(2);
					int expand = 1;
					while (i+1 + expand <= s.size() - 1 && i - expand >= 0 && s[i - expand] == s[i+1 + expand])
					{
						palindrome[i - expand].push_back(2 * expand + 2);
						expand++;
					}
				}
			}
		/*用回文子串组成原串*/
		vector<vector<string>> res;
		vector<string> cur;
		process(s, palindrome, cur, 0, res);

		return s.empty() ? vector<vector<string>>() : res;
	}

private:
	void process(string& s, const vector<vector<int>>& palindrome, vector<string>& cur, int curi, vector<vector<string>>& res)
	{
		if (curi == s.size())
		{
			res.push_back(cur);
			return;
		}

		for (int i = 0; i < palindrome[curi].size(); ++i)
		{
			cur.push_back(s.substr(curi, palindrome[curi][i]));
			process(s, palindrome, cur, curi + palindrome[curi][i], res);
			cur.pop_back();
		}
	}
};


/*----------------------------------------
		leetcode 133
 ---------------------------------------*/
class Node133 {
public:
	int val;
	vector<Node133*> neighbors;

	Node133() {
		val = 0;
		neighbors = vector<Node133*>();
	}

	Node133(int _val) {
		val = _val;
		neighbors = vector<Node133*>();
	}

	Node133(int _val, vector<Node133*> _neighbors) {
		val = _val;
		val = _val;
		neighbors = _neighbors;
	}
};
class Solution133 {
public:
	Node133* cloneGraph(Node133* node) {
		if (node == NULL)
			return NULL;
		unordered_map<Node133*, Node133*> map;
		stack<Node133*> s;
		s.push(node);
		while (!s.empty())
		{
			Node133* cur = s.top();
			s.pop();
			Node133* cloneNode = new Node133(cur->val);
			map[cur] = cloneNode;
			for (Node133* neighbor : cur->neighbors)
				if (map.find(neighbor) == map.end())
					s.push(neighbor);
		}

		for (auto iter = map.begin(); iter != map.end(); ++iter)
			for (Node133* neighbor : iter->first->neighbors)
				iter->second->neighbors.push_back(map[neighbor]);

		return map[node];
	}
};


/*----------------------------------------
		leetcode 134
 ---------------------------------------*/
class Solution134 {
public:
	int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
		/*-------------------------------------------------
		思路：原本需要O(n^2)复杂度遍历每种情况，
		可以优化成只需要O(n)一次遍历，依据为：
		从A站到不了B站，则A~B之间的任意一站都到不了B站
		-------------------------------------------------*/
		int start = 0;
		int cur = 1;
		int totalTank = 0;
		int curTank = 0;
		bool canReach = true;
		while (cur != start+gas.size()+1)
		{
			int last = (cur - 1 + gas.size()) % gas.size();
			curTank += (gas[last] - cost[last]);
			totalTank += (gas[last] - cost[last]);
			if (curTank < 0)
			{
				curTank = 0;
				totalTank = 0;
				start = cur % gas.size();
				canReach = false;
			}
			else
				canReach = true;

			cur++;
		}

		return canReach ? start : -1;
	}
};


/*----------------------------------------
		leetcode 136
 ---------------------------------------*/
class Solution136 {
public:
	int singleNumber(vector<int>& nums) {
		int XOR = 0;
		for (int item : nums)
			XOR = XOR ^ item;
		return XOR;
	}
};


/*----------------------------------------
		leetcode 137
 ---------------------------------------*/
class Solution137 {
public:
	int singleNumber(vector<int>& nums) {
		int once = 0;
		int twice = 0;
		for (int elem : nums)
		{
			once = ~twice & (once ^ elem);	//已经出现两次时，无条件置0；已经出现过一次时，1变0；没出现过时，0变1
			twice = ~once & (twice ^ elem); //当前是第一次，无条件置0；当前是第二次，0变1；当前是第三次，1变0
		}

		return once;
	}
};


/*----------------------------------------
		leetcode 138
 ---------------------------------------*/
class Node138 {
public:
	int val;
	Node138* next;
	Node138* random;

	Node138(int _val) {
		val = _val;
		next = NULL;
		random = NULL;
	}
};
class Solution138 {
public:
	Node138* copyRandomList(Node138* head) {
		Node138* cur = head;
		while (cur)
		{
			Node138* newNode = new Node138(cur->val);
			newNode->next = cur->next;
			cur->next = newNode;
			cur = newNode->next;
		}
		cur = head;
		while (cur)
		{
			cur->next->random = cur->random ? cur->random->next : NULL;
			cur = cur->next->next;
		}
		cur = head;
		Node138* ret = head ? head->next : NULL;
		while (cur)
		{
			Node138* tmp = cur->next;
			cur->next = tmp->next;
			cur = cur->next;
			tmp->next = cur ? cur->next : NULL;
		}

		return ret;
	}
};







/*bishi start*/

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
	Solution138 s;
	Node138* n1 = new Node138(1);
	Node138* n2 = new Node138(2);
	Node138* n3 = new Node138(3);
	n1->next = n2;
	n2->next = n3;
	n1->random = n3;
	n2->random = n2;
	n3->random = n1;
	Node138* ret = s.copyRandomList(n1);
	return 0;
}