<template>
    <n-tabs animated type="line">
        <n-tab-pane name="single" tab="Единичное распознавание">
            <div style="max-width: 768px; margin: auto">
                <n-form-item label="Примеры">
                    <div style="display:flex; gap: 8px">
                        <n-button secondary size="small" type="primary" @click="singleInput = 'Рубинштейна, 10А'">
                            Рубинштейна, 10А
                        </n-button>

                        <n-button secondary size="small" type="primary" @click="singleInput = 'Аптекарский, 18'">
                            Аптекарский, 18
                        </n-button>
                    </div>
                </n-form-item>

                <n-form-item label="Ввод">
                    <n-input v-model:value="singleInput"
                             placeholder="Введите адрес и отправьте запрос на распознавание"/>

                    <n-button style="margin-left: 12px;" type="primary" @click="singleRecognize">
                        Распознать
                    </n-button>
                </n-form-item>

                <n-form-item label="Вывод">
                    <n-input :loading="singleLoading" :value="singleOutput" placeholder="Нет результатов"
                             readonly
                             rows="20"
                             type="textarea"></n-input>
                </n-form-item>
            </div>
        </n-tab-pane>

        <n-tab-pane name="packet" tab="Пакетное распознавание">
            <div style="max-width: 998px; margin: auto; display:flex; gap: 24px">
                <n-form-item label="Ввод" style="width: 100%;">
                    <n-input v-model:value="packageInput" placeholder="Введите адреса (каждый с новой строки)" rows="20"
                             type="textarea"/>
                </n-form-item>

                <n-button style="margin: auto" type="primary" @click="packageRecognize" :loading="packageLoading">
                    →
                </n-button>

                <n-form-item label="Вывод" style="width: 100%;">
                    <n-collapse>
                        <n-collapse-item v-for="answer in packageOutput" :title="answer?.query">
                            <n-input :loading="packageLoading" :value="JSON.stringify(answer, null, 4)"
                                     placeholder="Нет результатов"
                                     readonly
                                     rows="20"
                                     type="textarea"></n-input>
                        </n-collapse-item>
                    </n-collapse>
                </n-form-item>
            </div>
        </n-tab-pane>

        <n-tab-pane name="autocomplete" tab="Автодополнение">
            <div style="max-width: 768px; margin: auto">
                <n-form-item label="Примеры">
                    <div style="display:flex; gap: 8px">
                        <n-button secondary size="small" type="primary" @click="autoCompleteInput = 'Лиговск'">
                            Лиговск..
                        </n-button>

                        <n-button secondary size="small" type="primary" @click="autoCompleteInput = 'Nevski'">
                            Nevski..
                        </n-button>
                    </div>
                </n-form-item>

                <n-form-item label="Ввод">
                    <n-input v-model:value="autoCompleteInput" placeholder="Начните вводить адрес"/>
                </n-form-item>

                <n-form-item label="Вывод">
                    <n-input :loading="autoCompleteLoading" :value="autoCompleteOutput" placeholder="Нет результатов"
                             readonly
                             rows="20"
                             type="textarea"></n-input>
                </n-form-item>
            </div>
        </n-tab-pane>
    </n-tabs>
</template>

<script lang="ts" setup>
import {ComputedRef, inject, Ref, ref, watchEffect} from "vue";
import {AxiosInstance as AxiosInstanceType} from "axios";

const axiosInstance: ComputedRef<AxiosInstanceType> = inject("axiosInstance", {} as ComputedRef<AxiosInstanceType>)


const singleLoading = ref(false)
const singleInput = ref("")
const singleOutput = ref("")

const singleRecognize = async () => {
  if (singleInput.value) {
    singleLoading.value = true
    axiosInstance.value.get("/search?query=" + singleInput.value).then((r) => {
      singleOutput.value = JSON.stringify(r.data, null, 4)
    }).finally(() => {
      singleLoading.value = false
    })
  }
}


const packageLoading = ref(false)
const packageInput = ref("")
const packageOutput = ref<any[]>([])

const packageRecognize = () => {
  if (packageInput.value) {
    packageLoading.value = true
    axiosInstance.value.post("/package_search", {values: packageInput.value.split("\n")}).then((r) => {
      packageOutput.value = r.data
    }).finally(() => {
      packageLoading.value = false
    })
  }
}


const autoCompleteLoading = ref(false)
const autoCompleteInput = ref("")
const autoCompleteOutput = ref("")
const autoCompleteDebounceTimer: Ref<number> = ref(0)


watchEffect(async () => {
  if (autoCompleteInput.value) {
    clearTimeout(autoCompleteDebounceTimer.value!)
    autoCompleteLoading.value = true

    autoCompleteDebounceTimer.value = setTimeout(() => {
      axiosInstance.value.get("/autocomplete?query=" + autoCompleteInput.value).then((r) => {
        autoCompleteOutput.value = r.data.join("\n")
      }).finally(() => {
        autoCompleteLoading.value = false
      })
    }, 300) as unknown as number
  }
})
</script>

<style scoped>

</style>