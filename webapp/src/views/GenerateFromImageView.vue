<template>
  <div>
    <h1>Locate face from picture</h1>
    <GenerateFromImageForm @generated="facesGenerated"/>
    <FacesModal :showModal="showModal" :imgs_bytes="imgs_bytes" :zs="zs" @close="closeModal"/>
  </div>
</template>
<script>
import { defineComponent } from 'vue';
import GenerateFromImageForm from '../components/GenerateFromImageForm.vue'
import FacesModal from '../components/FacesModal.vue'

// Components

export default defineComponent({
  name: 'GenerateFromImageView',
  components: {
    GenerateFromImageForm,
    FacesModal,
  },
  data() {
    return {
      showModal: false,
      zs: null,
      imgs_bytes: null
    }
  },
  methods: {
    closeModal(){
      this.showModal = false
    },
    facesGenerated(imgs_bytes, zs){
      this.showModal = true
      this.imgs_bytes = imgs_bytes
      this.zs = zs
    }
  }
});
</script>
