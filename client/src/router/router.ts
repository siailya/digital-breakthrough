import {createRouter, createWebHashHistory} from "vue-router";
import HomeView from "@views/HomeView.vue";
import FileProcessingView from "@views/FileProcessingView.vue";

export const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    {
      path: "/",
      component: HomeView
    },
    {
      path: "/files",
      component: FileProcessingView
    }
  ]
})