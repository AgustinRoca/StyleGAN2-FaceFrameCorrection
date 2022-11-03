<template>
  <div>
    <h1>Generate a transition between two faces</h1>
    <TransitionsForm @generated="transitionGenerated"/>
    <TransitionsModal :showModal="showModal" :imgs_bytes="imgs_bytes" :zs="zs" @close="closeModal"/>
  </div>
</template>
<script>
import { defineComponent } from 'vue';
import TransitionsForm from '../components/TransitionsForm.vue'
import TransitionsModal from '../components/TransitionsModal.vue'

// Components

export default defineComponent({
  name: 'TransitionsView',
  components: {
    TransitionsForm, TransitionsModal
  },
  data() {
    return {
      showModal: false,
      imgs_bytes: null,
      zs: null
    }
  },
  methods: {
    transitionGenerated(imgs_bytes, zs){
      this.showModal = true
      this.imgs_bytes = imgs_bytes
      this.zs = zs
    },
    closeModal(){
      this.showModal = false
    }
  }
});
</script>