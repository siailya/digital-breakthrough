import {createApp} from 'vue'
import App from '@/App.vue'
import {router} from "@/router/router.ts";
import "@/assets/style/animations.css"

createApp(App).use(router).mount('#app')
