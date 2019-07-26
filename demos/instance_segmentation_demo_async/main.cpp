// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine object_detection demo application
* \file object_detection_demo_ssd_async/main.cpp
* \example object_detection_demo_ssd_async/main.cpp
*/

#include <gflags/gflags.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

#include "instance_segmentation_demo_async.hpp"
#include <ext_list.hpp>

using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
       showUsage();
       showAvailableDevices();
       return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

void frameToBlob(const cv::Mat& frame,
                 InferRequest::Ptr& inferRequest,
                 const std::string& inputName) {
    if (FLAGS_auto_resize) {
        /* Just set input blob containing read image. Resize and layout conversion will be done automatically */
        inferRequest->SetBlob(inputName, wrapMat2Blob(frame));
    } else {
        /* Resize and copy data from the image to the input blob */
        Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
        matU8ToBlob<uint8_t>(frame, frameBlob);
    }
}


cv::Rect expand_box(const cv::Rect2f& box, float scale) {
    float w_half = box.width * .5;
    float h_half = box.height * .5;
    float x_c = box.x + w_half;
    float y_c = box.y + h_half;
    w_half *= scale;
    h_half *= scale;
    return cv::Rect(cv::Point(x_c - w_half, y_c - h_half),
                    cv::Point(x_c + w_half, y_c + h_half));
}



cv::Mat segm_postprocess(const cv::Rect2f& box, const cv::Mat& raw_cls_mask, int im_h, int im_w) {
    cv::Mat cls_mask;
    cv::copyMakeBorder(raw_cls_mask, cls_mask, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    float mask_height = cls_mask.rows;

    auto extended_box = cv::Rect(expand_box(box, mask_height / (mask_height - 2.0f)));
    int w = std::max(extended_box.width + 1, 1);
    int h = std::max(extended_box.height + 1, 1);
    int x0 = std::min(std::max(extended_box.tl().x, 0), im_w);
    int y0 = std::min(std::max(extended_box.tl().y, 0), im_h);
    int x1 = std::min(std::max(extended_box.br().x + 1, 0), im_w);
    int y1 = std::min(std::max(extended_box.br().y + 1, 0), im_h);

    cv::resize(cls_mask, cls_mask, cv::Size(w, h));
    auto mask = cls_mask > 0.5;
    // Put an object mask in an image mask.
    cv::Mat im_mask = cv::Mat::zeros(im_h, im_w, CV_8UC1);
    im_mask(cv::Rect(cv::Point(x0, y0), cv::Point(x1, y1))) =
        mask(cv::Rect(cv::Point(x0 - extended_box.x, y0 - extended_box.y),
                      cv::Point(x1 - extended_box.x, y1 - extended_box.y)));
    return im_mask;
}


std::vector<cv::Scalar> color_palette {{0, 113, 188},
                              {216, 82, 24},
                              {236, 176, 31},
                              {125, 46, 141},
                              {118, 171, 47},
                              {76, 189, 237},
                              {161, 19, 46},
                              {76, 76, 76},
                              {153, 153, 153},
                              {255, 0, 0},
                              {255, 127, 0},
                              {190, 190, 0},
                              {0, 255, 0},
                              {0, 0, 255},
                              {170, 0, 255},
                              {84, 84, 0},
                              {84, 170, 0},
                              {84, 255, 0},
                              {170, 84, 0},
                              {170, 170, 0},
                              {170, 255, 0},
                              {255, 84, 0},
                              {255, 170, 0},
                              {255, 255, 0},
                              {0, 84, 127},
                              {0, 170, 127},
                              {0, 255, 127},
                              {84, 0, 127},
                              {84, 84, 127},
                              {84, 170, 127},
                              {84, 255, 127},
                              {170, 0, 127},
                              {170, 84, 127},
                              {170, 170, 127},
                              {170, 255, 127},
                              {255, 0, 127},
                              {255, 84, 127},
                              {255, 170, 127},
                              {255, 255, 127},
                              {0, 84, 255},
                              {0, 170, 255},
                              {0, 255, 255},
                              {84, 0, 255},
                              {84, 84, 255},
                              {84, 170, 255},
                              {84, 255, 255},
                              {170, 0, 255},
                              {170, 84, 255},
                              {170, 170, 255},
                              {170, 255, 255},
                              {255, 0, 255},
                              {255, 84, 255},
                              {255, 170, 255},
                              {42, 0, 0},
                              {84, 0, 0},
                              {127, 0, 0},
                              {170, 0, 0},
                              {212, 0, 0},
                              {255, 0, 0},
                              {0, 42, 0},
                              {0, 84, 0},
                              {0, 127, 0},
                              {0, 170, 0},
                              {0, 212, 0},
                              {0, 255, 0},
                              {0, 0, 42},
                              {0, 0, 84},
                              {0, 0, 127},
                              {0, 0, 170},
                              {0, 0, 212},
                              {0, 0, 255},
                              {0, 0, 0},
                              {36, 36, 36},
                              {72, 72, 72},
                              {109, 109, 109},
                              {145, 145, 145},
                              {182, 182, 182},
                              {218, 218, 218},
                              {255, 255, 255}};


void overlay_mask(cv::Mat& image, const cv::Mat& mask, const int id=-1) {
    int color_idx = id % color_palette.size();
    auto mask_color = color_palette[color_idx];

    cv::Mat colored_mask(image.rows, image.cols, CV_8UC3);
    colored_mask.setTo(mask_color);
    auto segments_image = image.clone();
    cv::addWeighted(segments_image, 0.5, colored_mask, 0.5, 0.0, segments_image);
    segments_image.copyTo(image, mask);
}


struct Detection {
    cv::Mat mask;
    cv::Rect2f box;
    int label;
    float score;
    float area {0.0};
    int age {0};
    int id {-1};

    Detection(const cv::Rect2f& box, const cv::Mat& mask, int label, float score) {
        this->box = box;
        this->mask = mask;
        this->label = label;
        this->score = score;
        area = cv::countNonZero(mask);
        age = 0;
        id = -1;
    }
};


struct StaticIOUTracker{
    float iou_threshold;
    int age_threshold;
    int last_id;
    bool enable;
    std::vector<Detection> history;

    StaticIOUTracker(float iou_threshold=0.5f, int age_threshold=10, bool enable=true) {
        this->iou_threshold = iou_threshold;
        this->age_threshold = age_threshold;
        this->history.clear();
        this->last_id = 0;
        this->enable = enable;
    }

    cv::Mat affinity(std::vector<Detection>& candidates) {
        cv::Mat affinity_matrix = cv::Mat::zeros(candidates.size(), history.size(), CV_32F);
        for (size_t i = 0; i < candidates.size(); ++i) {
            const auto& mask = candidates[i].mask;
            candidates[i].area = cv::countNonZero(mask);
            const float area = candidates[i].area;
            const auto& box = candidates[i].box;
            for (size_t j = 0; j < history.size(); ++j) {
                const auto& hmask = history[j].mask;
                const float harea = history[j].area;
                if ((history[j].box & box).area() > 0) {
                    cv::Mat mask_intersection;
                    cv::bitwise_and(hmask, mask, mask_intersection);
                    float intersection_area = cv::countNonZero(mask_intersection);
                    float union_area = harea + area - intersection_area;
                    affinity_matrix.at<float>(i, j) = intersection_area / union_area;
                }
            }
        }
        return affinity_matrix;
    }

    void operator()(std::vector<Detection>& candidates) {
        if (!enable) {
            last_id = 0;
            for (auto& c: candidates) {
                c.id = last_id++;
            }
            return;
        }

        for (auto& h: history) {
            h.age++;
        }

        auto affinity_matrix = affinity(candidates);

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        for (size_t i = 0; i < std::min(candidates.size(), history.size()); ++i) {
            cv::minMaxLoc(affinity_matrix, &minVal, &maxVal, &minLoc, &maxLoc);
            auto iou = affinity_matrix.at<float>(maxLoc);
            assert(affinity_matrix.at<float>(maxLoc) == affinity_matrix.at<float>(maxLoc.y, maxLoc.x));
            if (iou > iou_threshold) {
                candidates[maxLoc.y].id = history[maxLoc.x].id;
                history[maxLoc.x] = candidates[maxLoc.y];
                history[maxLoc.x].age = 0;
                affinity_matrix.row(maxLoc.y).setTo(-1);
                affinity_matrix.col(maxLoc.x).setTo(-1);
            } else {
                break;
            }
        }

        for (auto& c: candidates) {
            if (c.id < 0) {
                c.id = last_id++;
                history.emplace_back(c);
            }
        }

        auto last = std::remove_if(history.begin(), history.end(),
                                   [&](const Detection& d) {return d.age > age_threshold;});
        history.erase(last, history.end());
    }

};


int main(int argc, char *argv[]) {
    try {
        /** This demo covers certain topology and cannot be generalized for any object detection **/
        std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        slog::info << "Reading input" << slog::endl;
        cv::VideoCapture cap;
        if (!((FLAGS_i == "cam") ? cap.open(0) : cap.open(FLAGS_i.c_str()))) {
            throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
        }
        const size_t width  = (size_t) cap.get(cv::CAP_PROP_FRAME_WIDTH);
        const size_t height = (size_t) cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        // read input (video) frame
        cv::Mat curr_frame;  cap >> curr_frame;
        cv::Mat next_frame;
        cv::resize(curr_frame, curr_frame, cv::Size(width, height));

        if (!cap.grab()) {
            throw std::logic_error("This demo supports only video (or camera) inputs !!! "
                                   "Failed getting next frame from the " + FLAGS_i);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load inference engine -------------------------------------
        slog::info << "Loading Inference Engine" << slog::endl;
        Core ie;

        slog::info << "Device info: " << slog::endl;
        std::cout << ie.GetVersions(FLAGS_d);

        /** Load extensions for the plugin **/

        /** Loading default extensions **/
        if (FLAGS_d.find("CPU") != std::string::npos) {
            /**
             * cpu_extensions library is compiled from "extension" folder containing
             * custom MKLDNNPlugin layer implementations. These layers are not supported
             * by mkldnn, but they can be useful for inferring custom topologies.
            **/
            ie.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>(), "CPU");
        }

        if (!FLAGS_l.empty()) {
            // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
            IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l.c_str());
            ie.AddExtension(extension_ptr, "CPU");
        }
        if (!FLAGS_c.empty()) {
            // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
            ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "GPU");
        }

        /** Per layer metrics **/
        if (FLAGS_pc) {
            ie.SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } });
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
        slog::info << "Loading network files" << slog::endl;
        CNNNetReader netReader;
        /** Read network model **/
        netReader.ReadNetwork(FLAGS_m);
        /** Set batch size to 1 **/
        slog::info << "Batch size is forced to  1." << slog::endl;
        netReader.getNetwork().setBatchSize(1);
        /** Extract model name and load it's weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        netReader.ReadWeights(binFileName);
        /** Read labels (if any)**/
        std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";
        std::vector<std::string> labels;
        std::ifstream inputFile(labelFileName);
        std::copy(std::istream_iterator<std::string>(inputFile),
                  std::istream_iterator<std::string>(),
                  std::back_inserter(labels));
        // -----------------------------------------------------------------------------------------------------

        /** SSD-based network should have one input and one output **/
        // --------------------------- 3. Configure input & output ---------------------------------------------
        // --------------------------- Prepare input blobs -----------------------------------------------------
        slog::info << "Checking that the inputs are as the demo expects" << slog::endl;
        InputsDataMap inputInfo(netReader.getNetwork().getInputsInfo());

        std::string imageInputName, imageInfoInputName;
        size_t netInputHeight, netInputWidth;

        for (const auto & inputInfoItem : inputInfo) {
            if (inputInfoItem.second->getTensorDesc().getDims().size() == 4) {  // first input contains images
                imageInputName = inputInfoItem.first;
                inputInfoItem.second->setPrecision(Precision::U8);
                if (FLAGS_auto_resize) {
                    inputInfoItem.second->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
                    inputInfoItem.second->getInputData()->setLayout(Layout::NHWC);
                } else {
                    inputInfoItem.second->getInputData()->setLayout(Layout::NCHW);
                }
                const TensorDesc& inputDesc = inputInfoItem.second->getTensorDesc();
                netInputHeight = getTensorHeight(inputDesc);
                netInputWidth = getTensorWidth(inputDesc);
            } else if (inputInfoItem.second->getTensorDesc().getDims().size() == 2) {  // second input contains image info
                imageInfoInputName = inputInfoItem.first;
                inputInfoItem.second->setPrecision(Precision::FP32);
            } else {
                throw std::logic_error("Unsupported " +
                                       std::to_string(inputInfoItem.second->getTensorDesc().getDims().size()) + "D "
                                       "input layer '" + inputInfoItem.first + "'. "
                                       "Only 2D and 4D input layers are supported");
            }
        }

        // --------------------------- Prepare output blobs -----------------------------------------------------
        slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
        OutputsDataMap outputInfo(netReader.getNetwork().getOutputsInfo());

        const SizeVector outputDims = outputInfo["raw_masks"]->getTensorDesc().getDims();
        const int maxDetectionsCount = outputDims[0];
        const int num_classes = outputDims[1];
        const int rawMaskHeight = outputDims[2];
        const int rawMaskWidth = outputDims[3];

        slog::info << maxDetectionsCount << ", " << num_classes << ", "
                   << rawMaskHeight << ", " << rawMaskWidth << slog::endl;

        if (static_cast<int>(labels.size()) != num_classes) {
            if (static_cast<int>(labels.size()) == (num_classes - 1))  // if network assumes default "background" class, having no label
                labels.insert(labels.begin(), "fake");
            else
                labels.clear();
        }

        outputInfo["boxes"]->setPrecision(Precision::FP32);
        outputInfo["boxes"]->setLayout(Layout::NC);
        outputInfo["scores"]->setPrecision(Precision::FP32);
        outputInfo["classes"]->setPrecision(Precision::FP32);
        outputInfo["raw_masks"]->setPrecision(Precision::FP32);
        outputInfo["raw_masks"]->setLayout(Layout::NCHW);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the device ------------------------------------------
        slog::info << "Loading model to the device" << slog::endl;
        ExecutableNetwork network = ie.LoadNetwork(netReader.getNetwork(), FLAGS_d);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        InferRequest::Ptr async_infer_request_curr = network.CreateInferRequestPtr();
        InferRequest::Ptr async_infer_request_next = network.CreateInferRequestPtr();

        /* it's enough just to set image info input (if used in the model) only once */
        if (!imageInfoInputName.empty()) {
            auto setImgInfoBlob = [&](const InferRequest::Ptr &inferReq) {
                auto blob = inferReq->GetBlob(imageInfoInputName);
                auto data = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
                data[0] = static_cast<float>(netInputHeight);  // height
                data[1] = static_cast<float>(netInputWidth);  // width
                data[2] = 1;
            };
            setImgInfoBlob(async_infer_request_curr);
            setImgInfoBlob(async_infer_request_next);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Do inference ---------------------------------------------------------
        slog::info << "Start inference " << slog::endl;
        StaticIOUTracker tracker;

        bool isLastFrame = false;
        bool isAsyncMode = true;  // execution is always started using SYNC mode
        bool isModeChanged = false;  // set to TRUE when execution mode is changed (SYNC<->ASYNC)

        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        auto total_t0 = std::chrono::high_resolution_clock::now();
        auto wallclock = std::chrono::high_resolution_clock::now();
        double ocv_decode_time = 0, ocv_render_time = 0;
        float fps = -1.0f;
        const float fps_alpha = 0.9f;

        std::cout << "To close the application, press 'CTRL+C' here or switch to the output window and press ESC key" << std::endl;
        std::cout << "To switch between sync/async modes, press TAB key in the output window" << std::endl;
        while (true) {
            auto t0 = std::chrono::high_resolution_clock::now();
            // Here is the first asynchronous point:
            // in the async mode we capture frame to populate the NEXT infer request
            // in the regular mode we capture frame to the CURRENT infer request
            if (!cap.read(next_frame)) {
                if (next_frame.empty()) {
                    isLastFrame = true;  // end of video file
                } else {
                    throw std::logic_error("Failed to get frame from cv::VideoCapture");
                }
            } else {
                cv::resize(next_frame, next_frame, cv::Size(width, height));
            }
            if (isAsyncMode) {
                if (isModeChanged) {
                    frameToBlob(curr_frame, async_infer_request_curr, imageInputName);
                }
                if (!isLastFrame) {
                    frameToBlob(next_frame, async_infer_request_next, imageInputName);
                }
            } else if (!isModeChanged) {
                frameToBlob(curr_frame, async_infer_request_curr, imageInputName);
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            ocv_decode_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            t0 = std::chrono::high_resolution_clock::now();
            // Main sync point:
            // in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
            // in the regular mode we start the CURRENT request and immediately wait for it's completion
            if (isAsyncMode) {
                if (isModeChanged) {
                    async_infer_request_curr->StartAsync();
                }
                if (!isLastFrame) {
                    async_infer_request_next->StartAsync();
                }
            } else if (!isModeChanged) {
                async_infer_request_curr->StartAsync();
            }

            if (OK == async_infer_request_curr->Wait(IInferRequest::WaitMode::RESULT_READY)) {
                t1 = std::chrono::high_resolution_clock::now();
                ms detection = std::chrono::duration_cast<ms>(t1 - t0);

                t0 = std::chrono::high_resolution_clock::now();
                ms wall = std::chrono::duration_cast<ms>(t0 - wallclock);
                wallclock = t0;

                fps = fps * fps_alpha + (1000.f / wall.count()) * (1.0f - fps_alpha);
                std::ostringstream out;
                out << "FPS: " << std::fixed << std::setprecision(1) << fps;
                cv::putText(curr_frame, out.str(), cv::Point2f(0, 25), cv::FONT_HERSHEY_TRIPLEX, 0.6, cv::Scalar(0, 0, 255));

                ocv_decode_time = ocv_render_time;
                ocv_render_time = ocv_decode_time;

                // ---------------------------Process output blobs--------------------------------------------------
                // Processing results of the CURRENT request
                const float *boxes = async_infer_request_curr->GetBlob("boxes")->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
                const float *classes = async_infer_request_curr->GetBlob("classes")->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
                const float *scores = async_infer_request_curr->GetBlob("scores")->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
                const float *raw_masks = async_infer_request_curr->GetBlob("raw_masks")->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

                const int objectSize = 4;
                const float scale_x = static_cast<float>(netInputWidth) / width;
                const float scale_y = static_cast<float>(netInputHeight) / height;
                std::vector<Detection> detections;

                for (int i = 0; i < maxDetectionsCount; i++) {
                    float image_id = 0;
                    if (image_id < 0) {
                        std::cout << "Only " << i << " proposals found" << std::endl;
                        break;
                    }

                    float confidence = scores[i];
                    auto label = static_cast<int>(classes[i]);
                    float xmin = boxes[i * objectSize + 0] / scale_x;
                    float ymin = boxes[i * objectSize + 1] / scale_y;
                    float xmax = boxes[i * objectSize + 2] / scale_x;
                    float ymax = boxes[i * objectSize + 3] / scale_y;

                    if (FLAGS_r) {
                        std::cout << "[" << i << "," << label << "] element, prob = " << confidence <<
                                  "    (" << xmin << "," << ymin << ")-(" << xmax << "," << ymax << ")"
                                  << ((confidence > FLAGS_t) ? " WILL BE RENDERED!" : "") << std::endl;
                    }

                    if (confidence > FLAGS_t) {
                        const cv::Mat raw_mask(rawMaskHeight, rawMaskWidth, CV_32FC1,
                            const_cast<float*>(&raw_masks[(i * num_classes + label) * rawMaskWidth * rawMaskHeight]));
                        cv::Rect2f box(cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax));
                        auto mask = segm_postprocess(box, raw_mask, height, width);
                        detections.emplace_back(box, mask, label, confidence);
                    }
                }

                tracker(detections);

                for (const auto& d: detections) {
                    overlay_mask(curr_frame, d.mask, d.id);
                    const auto& message = labels[d.label];
                    int baseline = 0;
                    auto textsize = cv::getTextSize(message, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                    cv::Point position((d.box.tl().x + d.box.br().x - textsize.width) / 2,
                                       (d.box.tl().y + d.box.br().y - textsize.height) / 2);
                    cv::putText(curr_frame, message, position, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                cv::Scalar(255, 255, 255), 1);
                }

            }
            cv::imshow("results", curr_frame);

            t1 = std::chrono::high_resolution_clock::now();
            ocv_render_time = std::chrono::duration_cast<ms>(t1 - t0).count();

            if (isLastFrame) {
                break;
            }

            if (isModeChanged) {
                isModeChanged = false;
            }

            // Final point:
            // in the truly Async mode we swap the NEXT and CURRENT requests for the next iteration
            curr_frame = next_frame;
            next_frame = cv::Mat();
            if (isAsyncMode) {
                async_infer_request_curr.swap(async_infer_request_next);
            }

            const int key = cv::waitKey(1);
            if (27 == key)  // Esc
                break;
            if (9 == key) {  // Tab
                isAsyncMode ^= true;
                isModeChanged = true;
            }
        }
        // -----------------------------------------------------------------------------------------------------
        auto total_t1 = std::chrono::high_resolution_clock::now();
        ms total = std::chrono::duration_cast<ms>(total_t1 - total_t0);
        std::cout << "Total Inference time: " << total.count() << std::endl;

        /** Show performace results **/
        if (FLAGS_pc) {
            printPerformanceCounts(*async_infer_request_curr, std::cout, getFullDeviceName(ie, FLAGS_d));
        }
    }
    catch (const std::exception& error) {
        std::cerr << "[ ERROR ] " << error.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}
