/*

Copyright (c) 2019 Ole-Christoffer Granmo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This code implements the Convolutional Tsetlin Machine from paper arXiv:1905.09688
https://arxiv.org/abs/1905.09688

*/

#include <stdio.h>
#include <stdlib.h>

#include "MultiClassConvolutionalTsetlinMachine.h"

/**************************************/
/*** The Convolutional Tsetlin Machine ***/
/**************************************/

/*** Initialize Tsetlin Machine ***/
struct MultiClassTsetlinMachine *CreateMultiClassTsetlinMachine(int number_of_classes, int number_of_clauses, int number_of_features, int number_of_patches, int number_of_ta_chunks, int number_of_state_bits, int T, double s, int boost_true_positive_feedback)
{

	struct MultiClassTsetlinMachine *mc_tm = NULL;

	mc_tm = (void *)malloc(sizeof(struct MultiClassTsetlinMachine));

	mc_tm->number_of_classes = number_of_classes;
	mc_tm->tsetlin_machines = (void *)malloc(sizeof(struct TsetlinMachine *)* number_of_classes);
	for (int i = 0; i < number_of_classes; i++) {
		mc_tm->tsetlin_machines[i] = CreateTsetlinMachine(number_of_clauses, number_of_features, number_of_patches, number_of_ta_chunks, number_of_state_bits, T, s, boost_true_positive_feedback);
	}
	
	mc_tm->number_of_patches = number_of_patches;

	mc_tm->number_of_ta_chunks = number_of_ta_chunks;

	mc_tm->number_of_state_bits = number_of_state_bits;

	return mc_tm;
}

void mc_tm_initialize(struct MultiClassTsetlinMachine *mc_tm)
{
	for (int i = 0; i < mc_tm->number_of_classes; i++) {
		tm_initialize(mc_tm->tsetlin_machines[i]);
	}
}


void mc_tm_destroy(struct MultiClassTsetlinMachine *mc_tm)
{
	for (int i = 0; i < mc_tm->number_of_classes; i++) {
		tm_destroy(mc_tm->tsetlin_machines[i]);

		free(mc_tm->tsetlin_machines[i]);
	}
	free(mc_tm->tsetlin_machines);
}

/***********************************/
/*** Predict classes of inputs X ***/
/***********************************/

void mc_tm_predict(struct MultiClassTsetlinMachine *mc_tm, unsigned int *X, int *y, int number_of_examples)
{
	int max_class;
	int max_class_sum;

	unsigned int step_size = mc_tm->number_of_patches * mc_tm->number_of_ta_chunks;

	unsigned int pos = 0;

	for (int l = 0; l < number_of_examples; l++) {
		// Identify class with largest output
		max_class_sum = tm_score(mc_tm->tsetlin_machines[0], &X[pos]);
		max_class = 0;
		for (int i = 1; i < mc_tm->number_of_classes; i++) {	
			int class_sum = tm_score(mc_tm->tsetlin_machines[i], &X[pos]);
			if (max_class_sum < class_sum) {
				max_class_sum = class_sum;
				max_class = i;
			}
		}

		y[l] = max_class;

		pos += step_size;
	}
	
	return;
}

//predict the class and print the clauses associated
void mc_tm_predict_and_print(struct MultiClassTsetlinMachine *mc_tm, unsigned int *X, int *y, int number_of_examples)
{
	printf("in mc_tm_predict_and_print");
	int max_class;
	int max_class_sum;
	int max_pos;

	unsigned int step_size = mc_tm->number_of_patches * mc_tm->number_of_ta_chunks;

	unsigned int pos = 0;

	for (int l = 0; l < number_of_examples; l++) {
		// Identify class with largest output
		max_class_sum = tm_score(mc_tm->tsetlin_machines[0], &X[pos]);
		max_class = 0;
		max_pos = 0;
		for (int i = 1; i < mc_tm->number_of_classes; i++) {	
			int class_sum = tm_score(mc_tm->tsetlin_machines[i], &X[pos]);
			if (max_class_sum < class_sum) {
				max_class_sum = class_sum;
				max_class = i;
				max_pos=pos;
			}
		}

		y[l] = max_class;
		printf("\nin mc_tm_predict_and_print first");
		tm_print_max_class(mc_tm->tsetlin_machines[max_class], &X[max_pos]);
		pos += step_size;
	}
	
	return;
}
		


/******************************************/
/*** Online Training of Tsetlin Machine ***/
/******************************************/

// The Tsetlin Machine can be trained incrementally, one training example at a time.
// Use this method directly for online and incremental training.

void mc_tm_update(struct MultiClassTsetlinMachine *mc_tm, unsigned int *Xi, int target_class)
{
	tm_update(mc_tm->tsetlin_machines[target_class], Xi, 1);

	// Randomly pick one of the other classes, for pairwise learning of class output 
	unsigned int negative_target_class = (unsigned int)mc_tm->number_of_classes * 1.0*rand()/((unsigned int)RAND_MAX + 1);
	while (negative_target_class == target_class) {
		negative_target_class = (unsigned int)mc_tm->number_of_classes * 1.0*rand()/((unsigned int)RAND_MAX + 1);
	}
	tm_update(mc_tm->tsetlin_machines[negative_target_class], Xi, 0);
}

/**********************************************/
/*** Batch Mode Training of Tsetlin Machine ***/
/**********************************************/

void mc_tm_fit(struct MultiClassTsetlinMachine *mc_tm, unsigned int *X, int *y, int number_of_examples, int epochs)
{
	unsigned int step_size = mc_tm->number_of_patches * mc_tm->number_of_ta_chunks;

	for (int epoch = 0; epoch < epochs; epoch++) {
		// Add shuffling here...
		unsigned int pos = 0;
		for (int i = 0; i < number_of_examples; i++) {
			mc_tm_update(mc_tm, &X[pos], y[i]);
			pos += step_size;
		}
	}
}

int mc_tm_ta_state(struct MultiClassTsetlinMachine *mc_tm, int class, int clause, int ta)
{
	return tm_ta_state(mc_tm->tsetlin_machines[class], clause, ta);
}

int mc_tm_ta_action(struct MultiClassTsetlinMachine *mc_tm, int class, int clause, int ta)
{
	return tm_ta_action(mc_tm->tsetlin_machines[class], clause, ta);
}

void mc_tm_clause_configuration(struct MultiClassTsetlinMachine *mc_tm, int class, int clause, unsigned int *clause_configuration)
{
	for (int k = 0; k < mc_tm->tsetlin_machines[class]->number_of_features; ++k) {
		clause_configuration[k] = tm_ta_action(mc_tm->tsetlin_machines[class], clause, k);
	}

	return;
}

int mc_tm_typeII_clause(struct MultiClassTsetlinMachine *mc_tm, int class, int clause, int ta)
{
	return tm_ta_typeII_clause(mc_tm->tsetlin_machines[class], clause, ta);
}

/*****************************************************/
/*** Storing and Loading of Tsetlin Machine State ****/
/*****************************************************/

void mc_tm_get_state(struct MultiClassTsetlinMachine *mc_tm, int class, unsigned int *ta_state)
{
	tm_get_state(mc_tm->tsetlin_machines[class], ta_state);
	
	return;
}

void mc_tm_set_state(struct MultiClassTsetlinMachine *mc_tm, int class, unsigned int *ta_state)
{
	tm_set_state(mc_tm->tsetlin_machines[class], ta_state);
	
	return;
}

/******************************************************************************/
/*** Clause Based Transformation of Input Examples for Multi-layer Learning ***/
/******************************************************************************/

void mc_tm_transform(struct MultiClassTsetlinMachine *mc_tm, unsigned int *X,  unsigned int *X_transformed, int invert, int number_of_examples)
{
	unsigned int step_size = mc_tm->number_of_patches * mc_tm->number_of_ta_chunks;

	unsigned int pos = 0;
	unsigned long transformed_feature = 0;
	for (int l = 0; l < number_of_examples; l++) {
		for (int i = 0; i < mc_tm->number_of_classes; i++) {	
			tm_score(mc_tm->tsetlin_machines[i], &X[pos]);

			for (int j = 0; j < mc_tm->tsetlin_machines[i]->number_of_clauses; ++j) {
				int clause_chunk = j / 32;
				int clause_pos = j % 32;

				int clause_output = (mc_tm->tsetlin_machines[i]->clause_output[clause_chunk] & (1 << clause_pos)) > 0;
	
				if (clause_output && !invert) {
					X_transformed[transformed_feature] = 1;
				} else if (!clause_output && invert) {
					X_transformed[transformed_feature] = 1;
				} else {
					X_transformed[transformed_feature] = 0;
				}

				transformed_feature++;
			} 
		}
		pos += step_size;
	}
	
	return;
}

