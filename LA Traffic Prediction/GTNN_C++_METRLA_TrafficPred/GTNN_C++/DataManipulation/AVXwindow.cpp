//
//#include "../DataManipulation/DataStore.h"
//
//
//
//float**& AVXwindow::get_input_segment(int tstart, int tend) {
//	for (int i = tstart; i < tend; i++) {
//		std::memcpy(AVX_Data::datasegment[i - tstart], AVX_Data::data[i], 3 * sizeof(float));
//	}
//	return datasegment;
//}
//
//float**& AVXwindow::get_true_segment(int tstart, int tend) {
//	for (int i = tstart; i < tend; i++) {
//		std::memcpy(AVX_Data::datasegment[i - tstart], AVX_Data::data_denoised[i], 3 * sizeof(float));
//	}
//	return datasegment;
//}
//
//float**& AVXwindow::get_input_window(int tstart, int tend, int dataCol) {
//	for (int i = tstart; i < tend; i++) {
//		AVX_Data::datawindow[0][i - tstart] = AVX_Data::data[i][dataCol];
//		
//	}
//	return datawindow;
//}
//
//float**& AVXwindow::get_true_window(int tstart, int tend, int dataCol) {
//	for (int i = tstart; i < tend; i++) {
//		AVX_Data::datawindow[0][i - tstart] = AVX_Data::data_denoised[i][dataCol];
//	}
//	return datawindow;
//}
//
//
//float**& AVXwindow::get_input_normalized_window(int tstart, int tend, int dataCol) {
//	get_input_window(tstart, tend, dataCol);
//	normalize_window();
//	return datawindow;
//}
//
//float**& AVXwindow::get_true_normalized_window(int tstart, int tend, int dataCol) {
//	get_true_window(tstart, tend, dataCol);
//	normalize_window();
//	return datawindow;
//}
//
//
////float**& AVXwindow::get_true_denormalized_window(int tstart, int tend, int dataCol) {
////	get_true_window(tstart, tend, dataCol);
////	denormalize_window();
////	return datawindow;
////}
////void AVXwindow::denormalize_window() {
////	for (int i = 0; i < AVX_Data::windowsize; i++) {
////		datawindow[0][i] *= normalize_var;
////	}
////}
//void AVXwindow::denormalize_window(float**& preds) {
//	for (int i = 0; i < AVX_Data::windowsize; i++) {
//		preds[0][i] *= normalize_var;
//	}
//}
//
//void AVXwindow::normalize_window() {
//	normalize_var = -99999.0;
//	for (int i = 0; i < AVX_Data::windowsize; i++) {
//		normalize_var = std::max(normalize_var, AVX_Data::datawindow[0][i]);
//	}
//	if (normalize_var < 0.1) normalize_var= 0.1;
//	for (int i = 0; i < AVX_Data::windowsize; i++) {
//		datawindow[0][i] /= normalize_var;
//	}
//}
//
