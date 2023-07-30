from rest_framework import serializers


class ClusteringSerializer(serializers.Serializer):
    input_file = serializers.FileField()

