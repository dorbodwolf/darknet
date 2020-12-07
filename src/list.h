/*
 * @Author: your name
 * @Date: 2020-11-16 15:52:14
 * @LastEditTime: 2020-12-05 15:24:14
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /darknet/src/list.h
 */
#ifndef LIST_H
#define LIST_H

// 链表上的节点
typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

// 双向链表
typedef struct list{
    int size; //链表上所有节点的个数
    node *front; //首节点
    node *back; //普通节点
} list;

#ifdef __cplusplus
extern "C" {
#endif
list *make_list();
int list_find(list *l, void *val);

void list_insert(list *, void *);

void **list_to_array(list *l);

void free_list_val(list *l);
void free_list(list *l);
void free_list_contents(list *l);
void free_list_contents_kvp(list *l);

#ifdef __cplusplus
}
#endif
#endif
