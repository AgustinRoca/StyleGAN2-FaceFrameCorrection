<template>
  <div>
    <h1>Change feature from image</h1>
    <FeaturesForm @generated="facesGenerated"/>
    <FacesModal :showModal="showModal" :imgs_bytes="[img_bytes]" :zs="[z]" @close="closeModal"/>
  </div>
</template>
<script>
import { defineComponent } from 'vue';
import FeaturesForm from '../components/FeaturesForm.vue'
import FacesModal from '../components/FacesModal.vue'

// Components

export default defineComponent({
  name: 'FeaturesView',
  components: {
    FeaturesForm,
    FacesModal,
  },
  data() {
    return {
      showModal: false,
      img_bytes: null,
      z: null
    }
  },
  methods: {
    facesGenerated(img_bytes, z){
      console.log(img_bytes)
      this.showModal = true
      this.img_bytes = img_bytes
      this.z = z
    },
    closeModal(){
      this.showModal = false
    }
  }
});
</script>
