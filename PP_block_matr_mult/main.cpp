#include<iostream>
#include <ctime>
#include<omp.h>
#include<cmath>
#include "mkl.h"
#include<queue>
using namespace std;
struct block {
	int i;
	int j;
	int k;
	
};

void mult_block(double *A, double *B, double *C, block q_block, int block_size, int n)
{
	for (int i = 0; i < block_size*block_size; i++)
		C[i] = 0;
	for (int j = 0; j < block_size; j++)
		for (int k = 0; k < block_size; k++)
			for (int i = 0; i < block_size; i++)
				C[j * block_size + i] += A[(q_block.j + j) * n + (q_block.k + k)] * B[(q_block.k + k) * n + (q_block.i + i)];
}

void matr_mult(double *A, double *B, double *C, int n)
{
	int i, j, k;
	double sum = 0;
#pragma omp parallel for  private(sum,i,j,k)
	for (i = 0; i < n; i++)
	{
		sum = 0;
		for (j = 0; j < n; j++)
		{
			for (k = 0; k < n; k++) {
				sum += A[i*n + k] * B[j + n*k];
			}
			C[i*n + j] = sum;
			sum = 0;
		}
	}
}

void block_matr_mult(double *A, double *B, double *C, int bSize, int n)
{
	for (int jk = 0; jk < n; jk += bSize)
		for (int kk = 0; kk < n; kk += bSize)
			for (int ik = 0; ik < n; ik += bSize)
				for (int j = 0; j < bSize; j++)
					for (int k = 0; k < bSize; k++)
						for (int i = 0; i < bSize; i++)
							C[(jk + j) * n + (ik + i)] += A[(jk + j) * n + (kk + k)] * B[(kk + k) * n + (ik + i)];
}

void sum_block_matr(double*A, double*res,block q_block, int b_size,int n)
{
	for (int j = 0; j < b_size; j++)
		for (int i = 0; i < b_size; i++)
		{
#pragma omp atomic
			res[j*n+q_block.j*n+q_block.i + i] += A[i + j*b_size];
		}
}

int cmp(double*a, double*b, int n)
{
	double eps = 0.0001;
	for (int i = 0; i < n*n; i++)
		if (abs(a[i] - b[i]) > eps)
		{
			cout << a[i] <<"    "<< b[i] << endl;
			cout << "Error i = " << i << endl;
			return 1;
		}
	return 0;
}

void view(double*a, int row, int col)
{
	for (int i = 0; i < col; i++) {
		for (int j = 0; j < row; j++)
			cout << a[i*col + j] << "  ";
		cout << endl;
	}
}
int main(int argc, char **argv)
{
	omp_set_num_threads(4);
	double *arr_one;
	double *arr_two;
	double *res;
	double * key;
	int n, i, j, k;
	double sum;
	double work_time;
	queue<block> qu_block,qu_block_copy;
	block tmp_block;
	int block_size;
	int num_threads = omp_get_max_threads();
	double** block_for_thread;
	cout << num_threads << endl;
	cout << "Input n ";
	if (argc > 1)
	{
		n = atoi(argv[1]);
		block_size = atoi(argv[2]);
	}
	cout << n << endl;

	arr_one = new double[n*n];
	arr_two = new double[n*n];
	res = new double[n*n];
	key = new double[n*n];

	for (int i = 0; i < n*n; i++)
	{
		arr_one[i] = 1;
		arr_two[i] = 1;
		key[i] = 0;
		res[i] = 0;
	}
	//==============


	block_for_thread = new double*[num_threads];
	for (int i = 0; i < num_threads; i++)
		block_for_thread[i] = new double[block_size*block_size];
	//===================
	work_time = omp_get_wtime();
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			for (int k = 0; k < n; k++)
				key[j*n + i] += arr_one[j*n + k] * arr_two[k*n + i];
	cout << "I J K No Parallel     Time:   " << omp_get_wtime() - work_time << " Sec." << endl;

	//===================
	
	work_time = omp_get_wtime();
	for (int jk = 0; jk < n; jk += block_size)
		for (int kk = 0; kk < n; kk += block_size)
			for (int ik = 0; ik < n; ik += block_size)
			{
				tmp_block.i = ik;
				tmp_block.j = jk;
				tmp_block.k = kk;
				qu_block.push(tmp_block);
			}
	qu_block_copy = qu_block;
	//--------
	omp_lock_t writelock;
	omp_init_lock(&writelock);

	omp_lock_t poplock;
	omp_init_lock(&poplock);
	//--------

	int id = omp_get_thread_num();
	work_time = omp_get_wtime();
#pragma omp parallel private (tmp_block)
	{
		id = omp_get_thread_num();
		while (!qu_block.empty())
		{
			omp_set_lock(&poplock);
			tmp_block = qu_block.front();
			qu_block.pop();
			omp_unset_lock(&poplock);
			mult_block(arr_one, arr_two, block_for_thread[id], tmp_block, block_size, n);
			omp_set_lock(&writelock);
			cout << id<<" ";
			sum_block_matr(block_for_thread[id], res, tmp_block, block_size, n);
			omp_unset_lock(&writelock);
		}
	}

cout << "Block queue parallel  Time:   " << omp_get_wtime() - work_time << " Sec." << endl;
	if (cmp(res, key, n))
		cout << "False" << endl;
	//====================
//========================
	for (int i = 0; i < n*n; i++)
		res[i] = 0;
	work_time = omp_get_wtime();
	block_matr_mult(arr_one, arr_two, res, block_size, n);
	cout << "Block queue           Time:   " << omp_get_wtime() - work_time << " Sec." << endl;
	if (cmp(res, key, n))
		cout << "False" << endl;

	delete[] arr_one;
	delete[] arr_two;
	delete[] res;
	delete[] key;

	omp_destroy_lock(&writelock);

}