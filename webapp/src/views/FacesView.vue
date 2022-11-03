<template>
  <div>
    <h1>Generate random faces</h1>
    <FacesForm @generated="facesGenerated"/>
    <FacesModal :showModal="showModal" :imgs_bytes="imgs_bytes" :zs="zs" @close="closeModal"/>
  </div>
</template>
<script>
import { defineComponent } from 'vue';
import FacesForm from '../components/FacesForm.vue'
import FacesModal from '../components/FacesModal.vue'

// Components

export default defineComponent({
  name: 'FacesView',
  components: {
    FacesForm,
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
    facesGenerated(imgs_bytes, zs){
      this.showModal = true
      this.zs = zs
      this.imgs_bytes = imgs_bytes
    },
    closeModal(){
      this.showModal = false
    }
  }
});
</script>
