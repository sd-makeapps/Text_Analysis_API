from django.http import JsonResponse
from rest_framework import viewsets
from .serializers import ClusteringSerializer
import pandas as pd
from gensim.models import Word2Vec
from janome.tokenizer import Tokenizer
from sklearn.cluster import KMeans
from transformers import pipeline
import random
from django.utils.datastructures import MultiValueDictKeyError


class AnalysisView(viewsets.ViewSet):
    serializer_class = ClusteringSerializer

    def analysis(self, request):
        try:
            input_file = request.FILES['input_file']
            if input_file.content_type == 'text/csv':
                df = pd.read_csv(input_file)

                # vectorize
                # 分かち書き（単語ごとに分割）
                def tokenize(text):
                    tokenizer = Tokenizer()
                    tokens = tokenizer.tokenize(text)
                    return [token.surface for token in tokens]

                # 分かち書きを実行
                try:
                    tokenized_texts = [tokenize(text) for text in df['text']]
                except MultiValueDictKeyError:
                    return JsonResponse({'msg': 'csv file format is invalid'})

                # Word2Vecモデルの学習
                model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=1)

                # テキストデータをベクトル化
                vectorized_texts = []
                for text in tokenized_texts:
                    text_vector = sum([model.wv[word] for word in text]) / len(text)
                    vectorized_texts.append(text_vector)

                # clustering
                num_clusters = 5
                clustering_model = KMeans(n_clusters=num_clusters)
                clustering_model.fit(vectorized_texts)

                # positive-negative judge
                tokenizer_name = "jarvisx17/japanese-sentiment-analysis"
                model_name = "jarvisx17/japanese-sentiment-analysis"
                pipe = pipeline(task="sentiment-analysis", model=model_name, tokenizer=tokenizer_name)

                dict1 = [[] for _ in range(num_clusters)]
                dict2 = []
                dict3 = []
                for i in range(len(df['text'])):
                    dict1[int(clustering_model.labels_[i])].append(
                        df['text'][i]
                    )
                    dict2.append(
                        {
                            'text': df['text'][i],
                            'label': int(clustering_model.labels_[i]),
                            'posi-nega': pipe(df['text'][i])[0]['label']
                        }
                    )
                for i in range(num_clusters):
                    dict3.append(
                        {
                            'label': i,
                            'summary': random.choice(dict1[i])
                        }
                    )

                response = {
                    'setting': dict3,
                    'data': dict2
                }

                return JsonResponse(response, safe=False, json_dumps_params={'ensure_ascii': False})
            else:
                return JsonResponse({'msg': 'please input csv file'})

        except MultiValueDictKeyError:
            return JsonResponse({'msg': 'please input any file'})
