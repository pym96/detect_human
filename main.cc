#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>


#include <string>
#include <iostream>

using namespace cv;

float cof_threshold = 0.8;
float nms_area_threshold = 0.5;

std::string MODEL_PATH = "/home/dan/learn/cv_service_inno/yolov8n-pose_openvino_model/yolov8n-pose.xml";
std::string DEVICE = "CPU";
cv::Mat img1;
cv::VideoCapture capture;

// 全局变量
std::vector<cv::Scalar> colors = { cv::Scalar(0, 0, 255) , cv::Scalar(0, 255, 0) , cv::Scalar(255, 0, 0) ,
                               cv::Scalar(255, 100, 50) , cv::Scalar(50, 100, 255) , cv::Scalar(255, 50, 100) };


std::vector<Scalar> colors_seg = { Scalar(255, 0, 0), Scalar(255, 0, 255), Scalar(170, 0, 255), Scalar(255, 0, 85),
                                   Scalar(255, 0, 170), Scalar(85, 255, 0), Scalar(255, 170, 0), Scalar(0, 255, 0),
                                   Scalar(255, 255, 0), Scalar(0, 255, 85), Scalar(170, 255, 0), Scalar(0, 85, 255),
                                   Scalar(0, 255, 170), Scalar(0, 0, 255), Scalar(0, 255, 255), Scalar(85, 0, 255) };

// 定义skeleton的连接关系以及color mappings
std::vector<std::vector<int>> skeleton = { {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13}, {6, 7},
                                          {6, 8}, {7, 9}, {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7} };

std::vector<cv::Scalar> posePalette = {
        cv::Scalar(255, 128, 0), cv::Scalar(255, 153, 51), cv::Scalar(255, 178, 102), cv::Scalar(230, 230, 0), cv::Scalar(255, 153, 255),
        cv::Scalar(153, 204, 255), cv::Scalar(255, 102, 255), cv::Scalar(255, 51, 255), cv::Scalar(102, 178, 255), cv::Scalar(51, 153, 255),
        cv::Scalar(255, 153, 153), cv::Scalar(255, 102, 102), cv::Scalar(255, 51, 51), cv::Scalar(153, 255, 153), cv::Scalar(102, 255, 102),
        cv::Scalar(51, 255, 51), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 0), cv::Scalar(255, 255, 255)
};

std::vector<int> limbColorIndices = { 9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16 };
std::vector<int> kptColorIndices = { 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9 };



void plot_keypoints(cv::Mat& image, const std::vector<std::vector<float>>& keypoints, const cv::Size& shape)
{

    int radius = 5;
    bool drawLines = true;

    if (keypoints.empty()) {
        return;
    }

    std::vector<cv::Scalar> limbColorPalette;
    std::vector<cv::Scalar> kptColorPalette;

    for (int index : limbColorIndices) {
        limbColorPalette.push_back(posePalette[index]);
    }

    for (int index : kptColorIndices) {
        kptColorPalette.push_back(posePalette[index]);
    }

    for (const auto& keypoint : keypoints) {
        bool isPose = keypoint.size() == 51;  // numKeypoints == 17 && keypoints[0].size() == 3;
        drawLines &= isPose;

        // draw points
        for (int i = 0; i < 17; i++) {
            int idx = i * 3;
            int x_coord = static_cast<int>(keypoint[idx]);
            int y_coord = static_cast<int>(keypoint[idx + 1]);

            if (x_coord % shape.width != 0 && y_coord % shape.height != 0) {
                if (keypoint.size() == 3) {
                    float conf = keypoint[2];
                    if (conf < 0.5) {
                        continue;
                    }
                }
                cv::Scalar color_k = isPose ? kptColorPalette[i] : cv::Scalar(0, 0,
                    255);  // Default to red if not in pose mode
                cv::circle(image, cv::Point(x_coord, y_coord), radius, color_k, -1, cv::LINE_AA);
            }
        }
        // draw lines
        if (drawLines) {
            for (int i = 0; i < skeleton.size(); i++) {
                const std::vector<int>& sk = skeleton[i];
                int idx1 = sk[0] - 1;
                int idx2 = sk[1] - 1;

                int idx1_x_pos = idx1 * 3;
                int idx2_x_pos = idx2 * 3;

                int x1 = static_cast<int>(keypoint[idx1_x_pos]);
                int y1 = static_cast<int>(keypoint[idx1_x_pos + 1]);
                int x2 = static_cast<int>(keypoint[idx2_x_pos]);
                int y2 = static_cast<int>(keypoint[idx2_x_pos + 1]);

                float conf1 = keypoint[idx1_x_pos + 2];
                float conf2 = keypoint[idx2_x_pos + 2];

                // Check confidence thresholds
                if (conf1 < 0.5 || conf2 < 0.5) {
                    continue;
                }

                // Check if positions are within bounds
                if (x1 % shape.width == 0 || y1 % shape.height == 0 || x1 < 0 || y1 < 0 ||
                    x2 % shape.width == 0 || y2 % shape.height == 0 || x2 < 0 || y2 < 0) {
                    continue;
                }

                // Draw a line between keypoints
                cv::Scalar color_limb = limbColorPalette[i];
                cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2), color_limb, 2, cv::LINE_AA);
            }
        }
    }
}

void letterbox(const cv::Mat& source, cv::Mat& result)
{
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
}

int main() {

    capture.open(0);
    if (!capture.isOpened()) {
        printf("could not load video data...\n");
        return -1;
    }

    //1.Create Runtime Core
    ov::Core core;

    //2.Compile the model
    ov::CompiledModel compiled_model = core.compile_model(MODEL_PATH, DEVICE);

    //3.Create inference request
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    while(true)
    {
        clock_t start_time, end_time;
        start_time = clock();

        capture >> img1;
        cv::Mat letterbox_img;
        letterbox(img1,letterbox_img);
        float scale = letterbox_img.size[0] / 640.0;
        cv::Mat blob = cv::dnn::blobFromImage(letterbox_img, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true);

        auto input_port = compiled_model.input();
        ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();

        //Get output
        auto output = infer_request.get_output_tensor(0);
        auto output_shape = output.get_shape();

        float* data = output.data<float>();
        cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
        cv::transpose(output_buffer, output_buffer); //[8400,23]
        cv::Mat dst = img1.clone();
        std::vector<int> class_ids;
        std::vector<float> class_scores;
        std::vector<cv::Rect> boxes;
        std::vector<std::vector<float>> objects_keypoints;

        //56: box[cx, cy, w, h] + Score + [17,3] keypoints
        for (int i = 0; i < output_buffer.rows; i++) {
            float class_score = output_buffer.at<float>(i, 4);
            if (class_score > cof_threshold) {
                class_scores.push_back(class_score);
                class_ids.push_back(0); //{0:"person"}
                float cx = output_buffer.at<float>(i, 0);
                float cy = output_buffer.at<float>(i, 1);
                float w = output_buffer.at<float>(i, 2);
                float h = output_buffer.at<float>(i, 3);
                // Get the box
                int left = int((cx - 0.5 * w) * scale);
                int top = int((cy - 0.5 * h) * scale);
                int width = int(w * scale);
                int height = int(h * scale);
                // Get the keypoints
                std::vector<float> keypoints;
                cv::Mat kpts = output_buffer.row(i).colRange(5, 56);
                for (int j = 0; j < 17; j++) {
                    float x = kpts.at<float>(0, j * 3 + 0) * scale;
                    float y = kpts.at<float>(0, j * 3 + 1) * scale;
                    float s = kpts.at<float>(0, j * 3 + 2);
                    keypoints.push_back(x);
                    keypoints.push_back(y);
                    keypoints.push_back(s);
                }

                boxes.push_back(cv::Rect(left, top, width, height));
                objects_keypoints.push_back(keypoints);
            }
        }
        
        //NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, class_scores, cof_threshold, nms_area_threshold, indices);

        dst = img1.clone();
        // Visualize the detection results
        for (size_t i = 0; i < indices.size(); i++) {
            int index = indices[i];
            // Draw bounding box
            cv::rectangle(dst, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
            std::string label = "Person:" + std::to_string(class_scores[index]).substr(0, 4);
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
            cv::Rect textBox(boxes[index].tl().x, boxes[index].tl().y - 15, textSize.width, textSize.height + 5);
            cv::rectangle(dst, textBox, cv::Scalar(0, 0, 255), cv::FILLED);
            cv::putText(dst, label, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
        }

        cv::Size shape = dst.size();
        plot_keypoints(dst, objects_keypoints, shape); // Assuming plot_keypoints function is defined elsewhere
        cv::imshow("dst_pose", dst);
        cv::waitKey(1); // Changed waitKey to avoid freezing
    }

    capture.release();

    return 0;
}
