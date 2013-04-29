#include <stdlib.h>
#include <pthread.h>

#include <stdio.h>

#include <queue.h>
#include <thread_pool.h>

void *
worker_thread_loop(void * arg);

ThreadPool *
new_thread_pool(int threads_num) {
    
    ThreadPool * pool = malloc(sizeof(ThreadPool));
    
    pool->tasks = new_queue();
    
    pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t cnd = PTHREAD_COND_INITIALIZER;
    pool->tasks_lock = mtx;
    pool->tasks_cond = cnd;
    
    pool->threads = calloc(threads_num, sizeof(pthread_t));
    pool->threads_num = threads_num;
    
    int i;
    for(i = 0; i < threads_num; i++) {
        pthread_create(&(pool->threads[i]), NULL, worker_thread_loop, pool);
    }
    
    return pool;
}

void *
worker_thread_loop(void * arg) {
    ThreadPool * pool = (ThreadPool *) arg;
    
    while(1) {
        pthread_mutex_lock(&(pool->tasks_lock));
        
        while(is_empty(pool->tasks))
            pthread_cond_wait(&(pool->tasks_cond), &(pool->tasks_lock));
        
        Task * task = (Task *) get(pool->tasks);
        
        pthread_mutex_unlock(&(pool->tasks_lock));
        
        if(task->type == TERMINATE) {
            break;
        }
        
        task->func(task->arg);
        
        pthread_mutex_lock(&(task->status_lock));
        task->status = DONE;
        pthread_mutex_unlock(&(task->status_lock));
        pthread_cond_signal(&(task->status_cond));
    }
    
    return NULL;
}

Task *
new_task(void (* func)(void *), void * arg) {
    Task * task = malloc(sizeof(Task));
    task->type = NORMAL;
    task->func = func;
    task->arg = arg;
    
    pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t cnd = PTHREAD_COND_INITIALIZER;
    
    task->status_lock = mtx;
    task->status_cond = cnd;
    
    return task;
}

void
execute_and_wait(Task ** tasks,
                 int count,
                 ThreadPool * pool) {
    
    int i;
    pthread_mutex_lock(&(pool->tasks_lock));
    for(i = 0; i < count; i++) {
        tasks[i]->status = ACTIVE;
        add(tasks[i], pool->tasks);
    }
    pthread_mutex_unlock(&(pool->tasks_lock));
    pthread_cond_broadcast(&(pool->tasks_cond));
    
    for(i = 0; i < count; i++) {
        pthread_mutex_lock(&(tasks[i]->status_lock));
        while(tasks[i]->status != DONE)
            pthread_cond_wait(&(tasks[i]->status_cond), &(tasks[i]->status_lock));
        pthread_mutex_unlock(&(tasks[i]->status_lock));
    }
}