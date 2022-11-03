<template>
  <div>
    <h1>Interchange styles between two faces</h1>
    <InterchangeForm @generated="faceGenerated"/>
    <FacesModal :showModal="showModal" :imgs_bytes="imgs_bytes" :zs="zs" @close="closeModal"/>
  </div>
</template>
<script>
import { defineComponent } from 'vue';
import InterchangeForm from '../components/InterchangeForm.vue'
import FacesModal from '../components/FacesModal.vue'

// Components

export default defineComponent({
  name: 'InterchangeView',
  components: {
    InterchangeForm, FacesModal
  },
  data() {
    return {
      showModal: false,
      imgs_bytes: null,
      zs: null
    }
  },
  methods: {
    faceGenerated(imgs_bytes, zs){
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