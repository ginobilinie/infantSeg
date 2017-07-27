#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MultiTaskSoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
  CHECK (this->layer_param_.multi_task_softmax_loss_param().has_task_id())
    << "Need Task ID.";


//puts("test1...........");
	
  if (this->layer_param_.multi_task_softmax_loss_param().weight_labels_size()) {
	int weight_labels_size = 
	  this->layer_param_.multi_task_softmax_loss_param().weight_labels_size();
    CHECK_EQ(weight_labels_size, 
	  this->layer_param_.multi_task_softmax_loss_param().labels_weight_size())
	  << "weight_labels_size should be equal to labels_weight_size";
//puts("test2..........");	
	int labels_weights_size = prob_.channels();
	labels_weights_ = new Dtype[labels_weights_size];
	 memset(labels_weights_, Dtype(1), sizeof(Dtype) * labels_weights_size);
        for (int i = 0; i < labels_weights_size; i ++)
        	labels_weights_[i] = Dtype(1.0);


	std::cout << "channels: " << labels_weights_size << ": " <<  labels_weights_[0] << ", " << labels_weights_[1] << std::endl;


//puts("test3..........");
	for (int i = 0; i < weight_labels_size; i ++) {
	  labels_weights_[this->layer_param_.multi_task_softmax_loss_param().weight_labels(i)]
	    = this->layer_param_.multi_task_softmax_loss_param().labels_weight(i);
	}
  }
  	
  task_id_ = this->layer_param_.multi_task_softmax_loss_param().task_id();
  
  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  normalize_ = this->layer_param_.loss_param().normalize();
}

template <typename Dtype>
void MultiTaskSoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  if (top.size() >= 2) {
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void MultiTaskSoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  
  int num = prob_.num();
  int dim = prob_.count() / num;
  int spatial_dim = prob_.height() * prob_.width();
  
  Dtype* label = new Dtype[num * spatial_dim];
  const Dtype* labels = bottom[1]->cpu_data();

   std::cout << "num: " << bottom[1]->num() << " ch: " << bottom[1]->channels() << " width: " << bottom[1]->width() << std::endl;
   std::cout << "1" << std::endl;

  int task_dim = bottom[1]->count() / num;


 // float percent0 = 0;

 
  for (int n = 0; n < num; n ++) {
	int offset1 = n * task_dim + task_id_ * spatial_dim;
	int offset2 = n * spatial_dim;
    for (int i = 0; i < spatial_dim; i ++) {
	  label[i+offset2] = labels[i + offset1];
		  if (labels[i + offset1] == 0) {
			percent0 ++;
		}
	}
  }

   std::cout << "task_id: " << task_id_ << " percent: " << percent0 / num / spatial_dim << std::endl;

   std::cout << "2" << std::endl;
  
  Dtype count = 0;
  Dtype loss = 0;

  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++) {
      const int label_value = static_cast<int>(label[i * spatial_dim + j]);

      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
    
      // std::cout << label_value << std::endl;

      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.channels());
	  
		loss -= labels_weights_[label_value] * log(std::max(prob_data[i * dim + label_value * spatial_dim + j],
                           Dtype(FLT_MIN)));
		count += labels_weights_[label_value];
    }
  }


  std::cout << "3" << std::endl;

  const int label_value = static_cast<int>(label[0]);
  std::cout << "task_id_" << task_id_ << " weight: " << labels_weights_[0] << "," << labels_weights_[1]  << std::endl; 




  if (normalize_) {
    top[0]->mutable_cpu_data()[0] = loss / count;
  } else {
    top[0]->mutable_cpu_data()[0] = loss / num;
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void MultiTaskSoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();

    int num = prob_.num();
    int dim = prob_.count() / num;
   
    int spatial_dim = prob_.height() * prob_.width();
    

    Dtype* label = new Dtype[num * spatial_dim];
    const Dtype* labels = bottom[1]->cpu_data();


    std::cout << "num: " << bottom[1]->num() << " ch: " << bottom[1]->channels() << " width: " << bottom[1]->width() << std::endl;
    int task_dim = bottom[1]->count() / num;

    for (int n = 0; n < num; n ++) {
          int offset1 = n * task_dim + task_id_ * spatial_dim;
          int offset2 = n * spatial_dim;
      for (int i = 0; i < spatial_dim; i ++) {
            label[i+offset2] = labels[i + offset1];
      }
    }



    int count = 0;
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        const int label_value = static_cast<int>(label[i * spatial_dim + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->channels(); ++c) {
            bottom_diff[i * dim + c * spatial_dim + j] = 0;
          }
        } else {
		  if (labels_weights_[label_value] == 1)
			bottom_diff[i * dim + label_value * spatial_dim + j] -= 1;
		  else {
			for (int c = 0; c < bottom[0]->channels(); c ++) {
				bottom_diff[i * dim + c * spatial_dim + j] *= 
				  labels_weights_[label_value];
			}	
			bottom_diff[i * dim + label_value * spatial_dim + j] -= 
			  labels_weights_[label_value];
		  }	
          ++count;
        }
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];

	int static tmpcnt = 0;

	if (tmpcnt ++ == 100) {
		tmpcnt = 0;
		printf("loss weight: %f\n", loss_weight);
	}

    if (normalize_) {
      caffe_scal(prob_.count(), loss_weight / count, bottom_diff);
    } else {
      caffe_scal(prob_.count(), loss_weight / num, bottom_diff);
    }
  }
}

INSTANTIATE_CLASS(MultiTaskSoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(MULTI_TASK_SOFTMAX_LOSS, MultiTaskSoftmaxWithLossLayer);

}  // namespace caffe
