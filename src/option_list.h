/*
 * @Author: your name
 * @Date: 2020-11-16 15:52:14
 * @LastEditTime: 2020-12-05 15:32:07
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /darknet/src/option_list.h
 */
#ifndef OPTION_LIST_H
#define OPTION_LIST_H
#include "darknet.h"
#include "list.h"

typedef struct{
    char *key; //解析后参数变量
    char *val; //参数值
    int used;
} kvp;

#ifdef __cplusplus
extern "C" {
#endif

list *read_data_cfg(char *filename);
int read_option(char *s, list *options);
void option_insert(list *l, char *key, char *val);
char *option_find(list *l, char *key);
char *option_find_str(list *l, char *key, char *def);
char *option_find_str_quiet(list *l, char *key, char *def);
int option_find_int(list *l, char *key, int def);
int option_find_int_quiet(list *l, char *key, int def);
float option_find_float(list *l, char *key, float def);
float option_find_float_quiet(list *l, char *key, float def);
void option_unused(list *l);

//typedef struct {
//	int classes;
//	char **names;
//} metadata;

//LIB_API metadata get_metadata(char *file);

#ifdef __cplusplus
}
#endif
#endif
