#include <stdlib.h>
#include <string.h>
#include "list.h"
#include "utils.h"
#include "option_list.h"

list *make_list()
{
    list* l = (list*)xmalloc(sizeof(list));
    l->size = 0;
    l->front = 0;
    l->back = 0;
    return l;
}

/*
void transfer_node(list *s, list *d, node *n)
{
    node *prev, *next;
    prev = n->prev;
    next = n->next;
    if(prev) prev->next = next;
    if(next) next->prev = prev;
    --s->size;
    if(s->front == n) s->front = next;
    if(s->back == n) s->back = prev;
}
*/

void *list_pop(list *l){
    if(!l->back) return 0;
    node *b = l->back;
    void *val = b->val;
    l->back = b->prev;
    if(l->back) l->back->next = 0;
    free(b);
    --l->size;

    return val;
}

/*
 * 将一个新变量插入到list链表的一个节点中
 * \param:l     链表
 * \param:val   插入的对象
 */
void list_insert(list *l, void *val)
{
    node* newnode = (node*)xmalloc(sizeof(node));
    newnode->val = val; //node变量的空指针指向val
    newnode->next = 0;

    if(!l->back){ // 在list的首节点插入
        l->front = newnode; // list的首节点指针指向newnode
        newnode->prev = 0; //首节点没有前向节点
    }else{ //插入list的普通节点
        l->back->next = newnode; // list的下一个节点指向newnode
        newnode->prev = l->back; // newnode的前向节点指向list当前节点
    }
    l->back = newnode; //list的当前节点更新为newnode
    ++l->size;
}

void free_node(node *n)
{
    node *next;
    while(n) {
        next = n->next;
        free(n);
        n = next;
    }
}

void free_list_val(list *l)
{
    node *n = l->front;
    node *next;
    while (n) {
        next = n->next;
        free(n->val);
        n = next;
    }
}

void free_list(list *l)
{
    free_node(l->front);
    free(l);
}

void free_list_contents(list *l)
{
    node *n = l->front;
    while(n){
        free(n->val);
        n = n->next;
    }
}

void free_list_contents_kvp(list *l)
{
    node *n = l->front;
    while (n) {
        kvp* p = (kvp*)n->val;
        free(p->key);
        free(n->val);
        n = n->next;
    }
}

void **list_to_array(list *l)
{
    void** a = (void**)xcalloc(l->size, sizeof(void*));
    int count = 0;
    node *n = l->front;
    while(n){
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}
