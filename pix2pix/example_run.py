

from model import train_pipeline
from validation_inference import run_validation, run_inference, run_batch_inference


def main():

    # run_validation(
    #     checkpoint_path="//path/to/unet_epoch100.pth",
    #     val_source_dir="//path/to/A/val",
    #     val_target_dir="//path/to/B/val",
    #     validation_output_dir="//path/to/validation_results"
    # )

    # run_inference(
    #     model_path="//path/to/unet_epoch100.pth",
    #     single_image_path="//path/to/INFERENCE/10001.png",
    #     single_output_path="//path/to/inference_results/10001_result.png",
    # )
    
    run_batch_inference(
        model_path="//path/to/unet_epoch100.pth",
        inference_input_dir="//path/to/INFERENCE",
        inference_output_dir="//path/to/inference_results",
    )

    # train_pipeline(
    #     source_dir="/path/to//A/train",
    #     target_dir="/path/to//B/train",
    #     save_dir="/path/to//checkpoints",
    #     epochs=100,
    #     batch_size=4,
    #     lr=1e-4
    # )




if __name__ == "__main__":
    main()