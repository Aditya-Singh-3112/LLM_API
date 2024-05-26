# your_app_name/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging

logger = logging.getLogger(__name__)
global_config = None
device_count = torch.cuda.device_count()
if device_count > 0:
    logger.debug("Select GPU device")
    device = torch.device("cuda")
else:
    logger.debug("Select CPU device")
    device = torch.device("cpu")

peft_model_id = "AdityaSingh312/Llama-7b-lamini-docs"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)

device_count = torch.cuda.device_count()
if device_count > 0:
    logger.debug("Select GPU device")
    device = torch.device("cuda")
else:
    logger.debug("Select CPU device")
    device = torch.device("cpu")

def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=150):
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=150)
    generated_text_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(text, '')
    # Strip the prompt

    return generated_text_answer

@api_view(['POST', 'GET'])
@csrf_exempt
def generate_text(request):
    if request.method == 'POST':
        try:
            data = request.data
            logger.debug(f"Received data: {data}")
            input_text = data.get('input_text', '')

            if not input_text:
                raise ValueError("Input text is empty")

            generated_text = inference(input_text, model, tokenizer)

            return Response({'generated_text': generated_text})
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return Response({'error': 'Error generating text'}, status=500)
    return Response({'error': 'Invalid request method'}, status=400)