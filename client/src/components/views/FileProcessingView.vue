<template>
    <div style="max-width: 768px; margin: auto;">
        <n-upload
                :action="axiosInstance.defaults.baseURL + '/file_process'"
                :custom-request="customRequest"
                accept=".txt,.json"
        >
            <n-upload-dragger>
                <div style="margin-bottom: 12px">
                    <n-icon :depth="3" size="48">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"
                             xmlns:xlink="http://www.w3.org/1999/xlink">
                            <g fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"
                               stroke-width="2">
                                <rect height="4" rx="2" width="18" x="3" y="4"></rect>
                                <path d="M5 8v10a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8"></path>
                                <path d="M10 12h4"></path>
                            </g>
                        </svg>
                    </n-icon>
                </div>
                <n-text style="font-size: 16px">
                    Нажмите или перетащите файл для обработки
                </n-text>
                <n-p depth="3" style="margin: 8px 0 0 0">
                    Принимаются .txt и .json файлы.
                    <br>
                    В txt-файлах каждый адрес должен быть написан с новой строки
                    <br>
                    В json-файлах должен быть массив из строк с адресами
                </n-p>
            </n-upload-dragger>
        </n-upload>

        <n-form-item label="Результаты распознавания">
            <n-input :loading="fileUploadLoading" :rows="24" :value="fileUploadOutput" readonly type="textarea"/>
        </n-form-item>
    </div>
</template>

<script lang="ts" setup>
import {ComputedRef, inject, ref} from "vue";
import {UploadCustomRequestOptions} from "naive-ui";
import axios, {AxiosInstance as AxiosInstanceType} from "axios";

const axiosInstance: ComputedRef<AxiosInstanceType> = inject("axiosInstance", {} as ComputedRef<AxiosInstanceType>)


const fileUploadOutput = ref("")
const fileUploadLoading = ref(false)

const customRequest = ({
                         file,
                         data,
                         action,
                         onFinish,
                         onError,
                       }: UploadCustomRequestOptions) => {
  fileUploadLoading.value = true
  const formData = new FormData()
  if (data) {
    Object.keys(data).forEach((key) => {
      formData.append(
        key,
        data[key as keyof UploadCustomRequestOptions['data']]
      )
    })
  }
  formData.append("file", file.file as File)

  axios.post(action as string, formData)
    .then(({data}) => {
      fileUploadOutput.value = JSON.stringify(data, null, 4)
      onFinish()
    })
    .catch(() => {
      onError()
    })
    .finally(() => {
      fileUploadLoading.value = false
    })
}
</script>

<style scoped>

</style>